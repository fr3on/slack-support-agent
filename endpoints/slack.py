"""Slack endpoint for the Support Agent plugin.

Receives Slack Events API webhooks (channel @mentions, direct messages, and
passive channel/group messages used to build thread context), forwards the
conversation to a linked Dify chat app, and posts the answer back to Slack.
"""

import json
import logging
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import requests
from dify_plugin import Endpoint
from dify_plugin.entities.model.message import SystemPromptMessage, UserPromptMessage
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from werkzeug import Request, Response

logger = logging.getLogger(__name__)

MENTION_PATTERN = re.compile(r"<@([A-Za-z0-9]+)>")
LOOSE_MENTION_PATTERN = re.compile(r"<@([^>]+)>")
SLACK_BLOCK_TEXT_LIMIT = 3000  # https://api.slack.com/reference/block-kit/composition-objects#text__fields
DM_CONVERSATION_ANCHOR = "dm-main"
# Message subtypes that are real user messages (a file upload with a caption,
# a thread reply also sent to the channel) rather than edits/deletes/joins.
CONVERSATION_SUBTYPES = ("file_share", "thread_broadcast")


class SlackMarkdownConverter:
    """Converts Markdown text into Slack's mrkdwn format.

    ref: https://github.com/fla9ua/markdown_to_mrkdwn
    """

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
        self.in_code_block = False
        self.table_replacements: Dict[str, str] = {}
        self.patterns: List[tuple] = [
            (re.compile(r"^(\s*)- \[([ ])\] (.+)", re.MULTILINE), r"\1• ☐ \3"),  # Unchecked task list
            (re.compile(r"^(\s*)- \[([xX])\] (.+)", re.MULTILINE), r"\1• ☑ \3"),  # Checked task list
            (re.compile(r"^(\s*)- (.+)", re.MULTILINE), r"\1• \2"),  # Unordered list
            (re.compile(r"^(\s*)(\d+)\. (.+)", re.MULTILINE), r"\1\2. \3"),  # Ordered list
            (re.compile(r"!\[.*?\]\((.+?)\)", re.MULTILINE), r"<\1>"),  # Images to URL
            (re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", re.MULTILINE), r"_\1_"),  # Italic
            (re.compile(r"^###### (.+)$", re.MULTILINE), r"*\1*"),  # H6 as bold
            (re.compile(r"^##### (.+)$", re.MULTILINE), r"*\1*"),  # H5 as bold
            (re.compile(r"^#### (.+)$", re.MULTILINE), r"*\1*"),  # H4 as bold
            (re.compile(r"^### (.+)$", re.MULTILINE), r"*\1*"),  # H3 as bold
            (re.compile(r"^## (.+)$", re.MULTILINE), r"*\1*"),  # H2 as bold
            (re.compile(r"^# (.+)$", re.MULTILINE), r"*\1*"),  # H1 as bold
            (re.compile(r"(^|\s)~\*\*(.+?)\*\*(\s|$)", re.MULTILINE), r"\1 *\2* \3"),  # Bold with space handling
            (re.compile(r"(?<!\*)\*\*(.+?)\*\*(?!\*)", re.MULTILINE), r"*\1*"),  # Bold
            (re.compile(r"__(.+?)__", re.MULTILINE), r"*\1*"),  # Underline as bold
            (re.compile(r"\[(.+?)\]\((.+?)\)", re.MULTILINE), r"<\2|\1>"),  # Links
            (re.compile(r"`(.+?)`", re.MULTILINE), r"`\1`"),  # Inline code
            (re.compile(r"^&gt; (.+)", re.MULTILINE), r"> \1"),  # Blockquote (">" is escaped first, see convert)
            (re.compile(r"^(---|\*\*\*|___)$", re.MULTILINE), r"──────────"),  # Horizontal line
            (re.compile(r"~~(.+?)~~", re.MULTILINE), r"~\1~"),  # Strikethrough
        ]
        self.triple_start = "%%BOLDITALIC_START%%"
        self.triple_end = "%%BOLDITALIC_END%%"

    def convert(self, markdown: str) -> str:
        if not markdown:
            return ""

        try:
            markdown = self._escape_slack_control_chars(markdown.strip())
            self.table_replacements = {}
            markdown = self._convert_tables(markdown)

            converted_lines = [self._convert_line(line) for line in markdown.split("\n")]
            result = "\n".join(converted_lines)

            for placeholder, table in self.table_replacements.items():
                result = result.replace(placeholder, table)

            return result.encode(self.encoding).decode(self.encoding)
        except Exception:
            return self._escape_slack_control_chars(markdown)

    @staticmethod
    def _escape_slack_control_chars(text: str) -> str:
        """Slack treats &, < and > as control characters: `<!channel>`,
        `<!here>` and `<@U123>` in model output would otherwise ping people
        (a prompt-injection route to mass pings), and a stray "<" can
        swallow text as a malformed link. Escaped up front; the converter
        then re-adds the < > it needs for real links."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _convert_tables(self, markdown: str) -> str:
        table_pattern = re.compile(
            r"^\|(.+)\|\s*$\n^\|[-:| ]+\|\s*$(\n^\|.+\|\s*$)*", re.MULTILINE
        )

        def convert_table(match: re.Match) -> str:
            original_table = match.group(0)
            table_lines = original_table.strip().split("\n")
            header_line = table_lines[0]
            data_lines = table_lines[2:] if len(table_lines) > 2 else []

            headers = [cell.strip() for cell in header_line.strip("|").split("|")]
            rows = [
                [cell.strip() for cell in line.strip("|").split("|")]
                for line in data_lines
            ]

            result = [" | ".join(f"*{header}*" for header in headers)]
            result.extend(" | ".join(row) for row in rows)

            placeholder = f"%%TABLE_PLACEHOLDER_{hash(original_table)}%%"
            self.table_replacements[placeholder] = "\n".join(result)
            return placeholder

        return table_pattern.sub(convert_table, markdown)

    def _convert_line(self, line: str) -> str:
        if line.startswith("%%TABLE_PLACEHOLDER_") and line.endswith("%%"):
            return line

        code_block_match = re.match(r"^```(\w*)$", line)
        if code_block_match:
            language = code_block_match.group(1)
            self.in_code_block = not self.in_code_block
            return f"```{language}" if self.in_code_block and language else "```"

        if self.in_code_block:
            return line

        line = re.sub(
            r"(?<!\*)\*\*\*([^*\n]+?)\*\*\*(?!\*)",
            lambda m: f"{self.triple_start}{m.group(1)}{self.triple_end}",
            line,
        )

        for pattern, replacement in self.patterns:
            line = pattern.sub(replacement, line)

        line = re.sub(
            re.escape(self.triple_start) + r"(.*?)" + re.escape(self.triple_end),
            r"*_\1_*",
            line,
            flags=re.MULTILINE,
        )

        return line.rstrip()


@dataclass
class ThreadMessage:
    """One message in a thread/DM, formatted for the linked Dify app."""

    role: str
    participant_id: str
    content: str
    ts: Optional[str] = None
    participant_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        # `ts` is only used internally to filter by context_scope - it's
        # never sent to the linked app.
        return {
            "role": self.role,
            "participant_id": self.participant_id,
            "content": self.content,
            "participant_name": self.participant_name,
        }


@dataclass
class RoutingContext:
    """Where an incoming Slack event routes to, and how the bot should reply.

    For a channel @mention, or a DM reply the user explicitly threaded,
    `raw_thread_ts` is a real Slack thread timestamp and every property below
    is simply that value (or the event's own ts for a brand-new thread).

    For a plain DM message - the common case, since Slack doesn't set
    thread_ts unless the user threads a reply - there's no natural thread to
    anchor to. Anchoring each message to its own ts would start a brand-new
    Dify conversation on every single message, so instead the whole DM
    channel shares one fixed conversation anchor, and replies post as normal
    top-level DM messages rather than a threaded reply under each message.
    """

    channel: str
    event_ts: Optional[str]
    is_dm: bool
    raw_thread_ts: Optional[str]

    @property
    def _is_freeform_dm(self) -> bool:
        return self.is_dm and not self.raw_thread_ts

    @property
    def conversation_ts(self) -> str:
        """Anchors our cache/conversation storage keys."""
        if self._is_freeform_dm:
            return DM_CONVERSATION_ANCHOR
        return self.raw_thread_ts or self.event_ts

    @property
    def reply_thread_ts(self) -> Optional[str]:
        """Value passed to Slack's chat_postMessage/conversations_replies."""
        if self._is_freeform_dm:
            return None
        return self.raw_thread_ts or self.event_ts

    @property
    def slack_thread_ts(self) -> Optional[str]:
        """The real Slack timestamp handed to the linked Dify app as input,
        for use by downstream plugins (e.g. Slack Post) - never our internal
        DM conversation anchor."""
        return self.raw_thread_ts or self.event_ts


class SlackEndpoint(Endpoint):
    CACHE_PREFIX = "thread-cache"
    CONVERSATION_PREFIX = "slack"
    CACHE_DURATION = 60 * 60 * 24  # 1 day
    MAX_RATE_LIMIT_WAIT = 20  # seconds

    # ---------------------------------------------------------------- cache

    def _cache_key(self, channel: str, conversation_ts: str) -> str:
        return f"{self.CACHE_PREFIX}-{channel}-{conversation_ts}"

    def _conversation_key(self, channel: str, conversation_ts: str) -> str:
        return f"{self.CONVERSATION_PREFIX}-{channel}-{conversation_ts}"

    def _load_cached_history(self, channel: str, conversation_ts: str) -> List[Dict[str, Any]]:
        key = self._cache_key(channel, conversation_ts)
        try:
            raw = self.session.storage.get(key)
        except Exception:
            return []
        if not raw:
            return []

        try:
            all_messages = json.loads(raw.decode("utf-8")).get("messages", [])
        except Exception:
            return []

        now = time.time()
        active = [m for m in all_messages if now - m.get("saved_at", now) < self.CACHE_DURATION]
        if len(active) != len(all_messages):
            self._write_cache(channel, conversation_ts, active)
        return active

    def _write_cache(self, channel: str, conversation_ts: str, messages: List[Dict[str, Any]]) -> None:
        try:
            self.session.storage.set(
                self._cache_key(channel, conversation_ts),
                json.dumps({"messages": messages}).encode("utf-8"),
            )
        except Exception as e:
            logger.warning("Failed to write thread cache (plugin storage full?): %s", e)

    def _append_thread_message(self, channel: str, conversation_ts: str, message: Mapping) -> None:
        messages = self._load_cached_history(channel, conversation_ts)
        msg = dict(message)
        # Slack can deliver the same message as both an app_mention and a
        # message event, so never store one ts twice.
        existing = next((m for m in messages if msg.get("ts") and m.get("ts") == msg.get("ts")), None)
        if existing is not None:
            if msg.get("to_agent"):
                existing["to_agent"] = True
                self._write_cache(channel, conversation_ts, messages)
            return
        msg["saved_at"] = time.time()
        messages.append(msg)
        self._write_cache(channel, conversation_ts, messages)

    def _get_conversation_id(self, channel: str, conversation_ts: str) -> Optional[str]:
        try:
            raw = self.session.storage.get(self._conversation_key(channel, conversation_ts))
        except Exception:
            return None
        return raw.decode("utf-8") if raw else None

    def _set_conversation_id(self, channel: str, conversation_ts: str, conversation_id: str) -> None:
        try:
            self.session.storage.set(
                self._conversation_key(channel, conversation_ts),
                conversation_id.encode("utf-8"),
            )
        except Exception as e:
            logger.warning("Failed to store conversation id (plugin storage full?): %s", e)

    # -------------------------------------------------------------- routing

    def _invoke(self, r: Request, values: Mapping, settings: Mapping) -> Response:
        if self._is_ignorable_retry(r, settings):
            return Response(status=200, response="ok")

        data = r.get_json(silent=True) or {}

        if data.get("type") == "url_verification":
            return Response(
                response=json.dumps({"challenge": data.get("challenge")}),
                status=200,
                content_type="application/json",
            )

        if data.get("type") != "event_callback":
            return Response(status=200, response="ok")

        event = data.get("event") or {}
        is_dm = self._is_dm_message(event)

        # Events subscribed "on behalf of users" (Slack app > Event
        # Subscriptions) also deliver DMs between a user who authorized the
        # app and *other people*. Those look identical to a DM with the bot,
        # so only handle DMs Slack delivered under the bot's own
        # authorization. This also stops a DM to the bot being answered twice
        # when both the bot and user subscriptions deliver it.
        if is_dm and not self._delivered_to_bot(data):
            return Response(status=200, response="ok")

        # Never answer other bots (or ourselves): two bots mentioning each
        # other would otherwise loop forever.
        if event.get("type") == "app_mention" and event.get("bot_id"):
            return Response(status=200, response="ok")

        if event.get("type") == "app_mention" or is_dm:
            return self._handle_reply(event, settings, is_dm)
        if event.get("type") == "message":
            return self._handle_passive_message(event, settings)
        return Response(status=200, response="ok")

    @staticmethod
    def _is_ignorable_retry(r: Request, settings: Mapping) -> bool:
        if settings.get("allow_retry"):
            return False
        retry_num = r.headers.get("X-Slack-Retry-Num")
        try:
            retried = retry_num is not None and int(retry_num) > 0
        except ValueError:
            retried = True
        return bool(r.headers.get("X-Slack-Retry-Reason") == "http_timeout" or retried)

    @staticmethod
    def _delivered_to_bot(data: Mapping) -> bool:
        """Whether Slack delivered this event under the bot's authorization
        (as opposed to a user's). If the payload carries no authorization
        info, assume it did rather than drop real DMs."""
        authorizations = data.get("authorizations")
        if not isinstance(authorizations, list) or not authorizations:
            return True
        return any(a.get("is_bot") for a in authorizations if isinstance(a, Mapping))

    @staticmethod
    def _is_dm_message(event: Mapping) -> bool:
        """Slack sends type="message" with channel_type="im" for a DM (no
        app_mention fires in a DM, since there's no one else to @-mention).
        Guard against bot_id/subtype so we never reply to our own posts or to
        edits/deletes - either would otherwise create a reply loop."""
        return bool(
            event.get("type") == "message"
            and event.get("channel_type") == "im"
            and not event.get("bot_id")
            and event.get("subtype") in (None, *CONVERSATION_SUBTYPES)
            and (event.get("text") or event.get("files"))
        )

    @staticmethod
    def _event_to_cache_entry(event: Mapping, to_agent: bool = False) -> Dict[str, Any]:
        entry = {
            "ts": event.get("ts"),
            "text": event.get("text", ""),
            "user": event.get("user"),
            "bot_id": event.get("bot_id"),
        }
        if to_agent:
            # Set for messages that were addressed to the agent (an @mention
            # or a DM), so they survive the "agent conversation only" filter.
            entry["to_agent"] = True
        return entry

    # ------------------------------------------------- passive caching path

    def _handle_passive_message(self, event: Mapping, settings: Mapping) -> Response:
        """Messages that aren't a mention or a DM (e.g. other people replying
        in a channel thread the bot is already tracking) are cached silently,
        without invoking the linked app, so full thread context is available
        next time the bot is mentioned in that thread."""
        # Edits, deletions, joins/leaves and other housekeeping events are not
        # conversation messages - caching them would insert blank/garbage
        # entries into the history.
        if event.get("subtype") not in (None, *CONVERSATION_SUBTYPES) or not event.get("text"):
            return Response(status=200, response="ok")

        # With "exclude user-to-user messages" on, human chatter is neither
        # cached nor forwarded: messages addressed to the agent arrive via the
        # mention/DM path and the agent's own replies are cached when posted.
        if settings.get("exclude_user_to_user_messages", False):
            return Response(status=200, response="ok")

        channel = event.get("channel", "")
        thread_ts = event.get("thread_ts") or event.get("ts")
        if self._is_thread_recognized(channel, thread_ts):
            self._append_thread_message(channel, thread_ts, self._event_to_cache_entry(event))
        return Response(status=200, response="ok")

    def _is_thread_recognized(self, channel: str, thread_ts: str) -> bool:
        for key in (
            self._conversation_key(channel, thread_ts),
            self._cache_key(channel, thread_ts),
        ):
            try:
                if self.session.storage.get(key):
                    return True
            except Exception:
                pass
        return False

    # ----------------------------------------------------- mention -> reply

    def _handle_reply(self, event: Mapping, settings: Mapping, is_dm: bool) -> Response:
        ctx = RoutingContext(
            channel=event.get("channel", ""),
            event_ts=event.get("ts"),
            is_dm=is_dm,
            raw_thread_ts=event.get("thread_ts"),
        )
        client = WebClient(token=settings.get("bot_token"))
        message_text = self._extract_message_text(event)
        if not message_text.strip() and event.get("files"):
            message_text = "(The user attached file(s) without any text.)"

        # Slack can deliver the same message twice without retry headers
        # (e.g. through two event subscriptions). If we've already taken this
        # one, don't answer twice - unless retries are explicitly allowed.
        history = self._load_cached_history(ctx.channel, ctx.conversation_ts)
        if not settings.get("allow_retry") and any(
            m.get("ts") == event.get("ts") and m.get("to_agent") for m in history
        ):
            return Response(status=200, response="ok")

        blocked = self._enforce_allowed_channel(client, ctx, settings, is_dm)
        if blocked is not None:
            return blocked

        # Captured before the pre-append below, so we can tell whether this
        # thread had any history *before* this message - see
        # _fetch_thread_messages for why that distinction matters.
        had_cached_history = bool(history)
        self._append_thread_message(
            ctx.channel, ctx.conversation_ts, self._event_to_cache_entry(event, to_agent=True)
        )

        try:
            return self._respond(client, ctx, settings, event, message_text, had_cached_history)
        except Exception as e:
            return self._handle_invoke_error(client, ctx, settings, e)

    @staticmethod
    def _extract_message_text(event: Mapping) -> str:
        text = event.get("text", "")
        if event.get("type") == "app_mention":
            # Remove the leading bot mention, e.g. "<@U123> what's up" -> "what's up".
            # DMs carry no leading mention, so nothing to strip there.
            return re.sub(r"^<@[^>]+>\s*", "", text)
        return text

    def _enforce_allowed_channel(
        self, client: WebClient, ctx: RoutingContext, settings: Mapping, is_dm: bool
    ) -> Optional[Response]:
        """Returns a short-circuit Response if this channel isn't allowed (or
        the check itself failed), else None to continue. DMs have no channel
        name to compare against - the restriction is a channel-scoping
        feature, so it's skipped for DMs entirely."""
        allowed_channel = settings.get("allowed_channel", "").strip()
        if not allowed_channel or is_dm:
            return None

        try:
            channel_info = client.conversations_info(channel=ctx.channel)
            actual_channel = f"#{channel_info['channel']['name']}"
        except SlackApiError as e:
            logger.warning("Error getting channel info: %s", e)
            self._safe_post(client, ctx, f"Failed to retrieve channel info. SlackApiError: {e}")
            return Response(status=200, response="ok", content_type="text/plain")
        except Exception as e:
            logger.warning("Unexpected error getting channel info: %s", e)
            self._safe_post(
                client, ctx, f"An unexpected error occurred while retrieving channel info. Error: {e}"
            )
            return Response(status=200, response="ok", content_type="text/plain")

        if actual_channel.lower() == "#" + allowed_channel.lstrip("#").lower():
            return None

        self._safe_post(client, ctx, f"Current channel: {actual_channel} is not allowed.")
        return Response(status=200, response="ok", content_type="text/plain")

    @staticmethod
    def _safe_post(client: WebClient, ctx: RoutingContext, text: str) -> None:
        try:
            client.chat_postMessage(channel=ctx.channel, thread_ts=ctx.reply_thread_ts, text=text)
        except Exception as e:
            logger.warning("Failed to post notice to Slack: %s", e)

    # ------------------------------------------------------- app invocation

    def _respond(
        self,
        client: WebClient,
        ctx: RoutingContext,
        settings: Mapping,
        event: Mapping,
        message_text: str,
        had_cached_history: bool,
    ) -> Response:
        conversation_id = self._get_conversation_id(ctx.channel, ctx.conversation_ts)

        raw_messages = self._fetch_thread_messages(client, ctx, had_cached_history)
        # Decided from the unfiltered thread: "first message" means the thread
        # has had no earlier messages at all, whatever the filter keeps.
        is_first_message = len(raw_messages) == 1
        if settings.get("exclude_user_to_user_messages", False) and not ctx.is_dm:
            raw_messages = self._filter_agent_conversation(client, ctx, raw_messages)
        user_display_names = self._resolve_display_names(client, raw_messages)
        own_bot_id = self._get_bot_identity(client).get("bot_id")
        thread_history, _ = self._build_thread_history(raw_messages, user_display_names, own_bot_id)
        thread_history = self._apply_context_scope(
            thread_history, settings.get("context_scope", "full_thread"), ctx.event_ts
        )
        if settings.get("summarize_thread_history", False):
            thread_history = self._summarize_thread_history(thread_history, settings.get("summarization_model"))

        uploaded_files = self._upload_slack_files(
            client, ctx, settings.get("bot_token"), event.get("files", [])
        )

        invoke_params: Dict[str, Any] = {
            "app_id": settings["app"]["app_id"],
            "query": self._substitute_mentions(message_text, user_display_names),
            "inputs": self._build_invoke_inputs(ctx, thread_history, user_display_names, uploaded_files),
            "response_mode": "blocking",
            # Attributes the conversation to this Slack channel/DM in Dify's
            # own Logs and Messages API, instead of every Slack conversation
            # falling back to one shared anonymous end user.
            "user": f"slack-{ctx.channel}",
        }
        if conversation_id is not None:
            invoke_params["conversation_id"] = conversation_id

        try:
            response = self.session.app.chat.invoke(**invoke_params)
        except Exception as e:
            # The stored Dify conversation may have been deleted; without this
            # every later message in the thread would fail forever. Start a
            # fresh conversation once instead.
            msg = str(e).lower()
            if conversation_id is not None and "conversation" in msg and ("not exist" in msg or "not found" in msg):
                logger.warning("Dify conversation %s no longer exists, starting a new one", conversation_id)
                invoke_params.pop("conversation_id", None)
                response = self.session.app.chat.invoke(**invoke_params)
            else:
                raise
        answer = (response.get("answer") or "").strip() or "Sorry, I couldn't come up with an answer to that."

        new_conversation_id = response.get("conversation_id")
        if new_conversation_id:
            self._set_conversation_id(ctx.channel, ctx.conversation_ts, new_conversation_id)

        try:
            return self._post_answer(client, ctx, answer, settings, is_first_message)
        except SlackApiError as e:
            return Response(
                status=200, response=f"Error sending message to Slack: {e}", content_type="text/plain"
            )

    def _fetch_thread_messages(
        self, client: WebClient, ctx: RoutingContext, had_cached_history: bool
    ) -> List[Dict[str, Any]]:
        """Raw Slack messages for this thread/DM.

        `had_cached_history` must reflect whether the cache had anything in it
        *before* the current message was appended - checking cache emptiness
        after that append would always find at least the just-added message,
        permanently hiding whether this thread's older context was ever
        fetched. That matters because our local cache entries expire after
        CACHE_DURATION (24h): a thread that's been quiet for a day looks
        "empty" locally even though Slack still has the full history, and
        without this distinction we'd never re-fetch it - the linked app
        would silently see only the newest message and lose all older
        context. Only falls back to Slack's API when we have a real thread ts
        - the synthetic DM anchor isn't one.
        """
        messages = self._load_cached_history(ctx.channel, ctx.conversation_ts)
        # A message that isn't inside an existing Slack thread has no earlier
        # history to fetch - skip the API call (thread-history calls are
        # heavily rate limited).
        if had_cached_history or not ctx.raw_thread_ts:
            messages.sort(key=lambda m: float(m.get("ts") or 0))
            return messages

        fetched = self._fetch_replies_with_retry(client, ctx)
        return self._merge_and_cache_messages(ctx.channel, ctx.conversation_ts, messages, fetched)

    def _merge_and_cache_messages(
        self,
        channel: str,
        conversation_ts: str,
        existing: List[Dict[str, Any]],
        fetched: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merges freshly fetched Slack messages into the cached list,
        de-duplicating by `ts` (Slack's fetch includes the current triggering
        message too, which is already in `existing`), sorts chronologically,
        and persists the result as the new cache contents."""
        merged = list(existing)
        seen_ts = {m.get("ts") for m in merged}
        for m in fetched:
            if m.get("ts") not in seen_ts:
                merged.append(m)
                seen_ts.add(m.get("ts"))
        merged.sort(key=lambda m: float(m.get("ts") or 0))

        now = time.time()
        for m in merged:
            m.setdefault("saved_at", now)
        self._write_cache(channel, conversation_ts, merged)
        return merged

    def _fetch_replies_with_retry(self, client: WebClient, ctx: RoutingContext) -> List[Dict[str, Any]]:
        try:
            return client.conversations_replies(channel=ctx.channel, ts=ctx.reply_thread_ts).get("messages", [])
        except SlackApiError as e:
            if e.response.get("error") != "ratelimited":
                logger.warning("Error getting thread history: %s", e)
                return []

            try:
                retry_after = int((getattr(e.response, "headers", None) or {}).get("Retry-After", 30))
            except (TypeError, ValueError):
                retry_after = 30
            if retry_after > self.MAX_RATE_LIMIT_WAIT:
                # Sleeping this long would outlast the Slack/Dify request;
                # answer with what we have cached instead.
                logger.warning("Slack rate limit wait of %ss too long, skipping thread fetch", retry_after)
                return []
            self._safe_post(
                client, ctx, f"Rate limit reached when retrieving thread. Retrying in {retry_after} seconds..."
            )
            time.sleep(retry_after)
            try:
                return client.conversations_replies(channel=ctx.channel, ts=ctx.reply_thread_ts).get("messages", [])
            except SlackApiError as retry_error:
                logger.warning("Error getting thread history after retry: %s", retry_error)
                return []

    BOT_IDENTITY_KEY = "bot-identity"

    def _get_bot_identity(self, client: WebClient) -> Dict[str, Optional[str]]:
        """This bot's Slack user id and bot id, cached in plugin storage so
        auth.test is only called once."""
        try:
            raw = self.session.storage.get(self.BOT_IDENTITY_KEY)
            if raw:
                return json.loads(raw.decode("utf-8"))
        except Exception:
            pass
        try:
            info = client.auth_test()
            identity = {"user_id": info.get("user_id"), "bot_id": info.get("bot_id")}
        except SlackApiError as e:
            logger.warning("auth.test failed, cannot identify the bot: %s", e)
            return {"user_id": None, "bot_id": None}
        try:
            self.session.storage.set(self.BOT_IDENTITY_KEY, json.dumps(identity).encode("utf-8"))
        except Exception:
            pass
        return identity

    def _filter_agent_conversation(
        self, client: WebClient, ctx: RoutingContext, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Keeps only the messages that are part of the conversation with the
        agent, dropping messages the channel's people sent to each other:
        the thread's root message (the original question), the agent's own
        replies, and any message that @mentions the agent. If the bot's
        identity can't be determined, returns everything unfiltered rather
        than risk dropping the user's actual question."""
        identity = self._get_bot_identity(client)
        bot_user_id, bot_id = identity.get("user_id"), identity.get("bot_id")
        if not bot_user_id:
            return messages

        mention = f"<@{bot_user_id}>"
        root_ts = ctx.raw_thread_ts or ctx.event_ts
        return [
            m
            for m in messages
            if m.get("ts") == root_ts
            or m.get("to_agent")
            or m.get("user") == bot_user_id
            or (bot_id and m.get("bot_id") == bot_id)
            or mention in (m.get("text") or "")
        ]

    def _resolve_display_names(self, client: WebClient, messages: List[Dict[str, Any]]) -> Dict[str, str]:
        user_ids: List[str] = []
        for msg in messages:
            user_id = msg.get("user", "unknown")
            if user_id != "unknown" and user_id not in user_ids:
                user_ids.append(user_id)
            for mentioned_id in LOOSE_MENTION_PATTERN.findall(msg.get("text") or ""):
                if mentioned_id not in user_ids:
                    user_ids.append(mentioned_id)

        display_names: Dict[str, str] = {}
        for user_id in user_ids:
            # One unresolvable id (deleted user, "U123|name" legacy mention,
            # a bot id) must not stop the rest from resolving.
            try:
                info = client.users_info(user=user_id.split("|")[0]).get("user", {})
            except SlackApiError as e:
                logger.warning("Error getting user info for %s: %s", user_id, e)
                continue
            name, real_name = info.get("name", ""), info.get("real_name", "")
            display_names[user_id] = f"{real_name} ({name})" if (name and real_name) else (real_name or name)
        return display_names

    @staticmethod
    def _is_own_message(msg: Mapping, own_bot_id: Optional[str]) -> bool:
        """Only this bot's own posts are the assistant's turns; other bots
        (Jira, GitHub, ...) in the same thread are just other participants.
        If our identity is unknown, fall back to treating any bot as us."""
        if not msg.get("bot_id"):
            return False
        return own_bot_id is None or msg["bot_id"] == own_bot_id

    def _build_thread_history(
        self,
        messages: List[Dict[str, Any]],
        user_display_names: Dict[str, str],
        own_bot_id: Optional[str] = None,
    ) -> tuple:
        thread_history = [
            ThreadMessage(
                role="assistant" if self._is_own_message(msg, own_bot_id) else "user",
                participant_id=(participant_id := msg.get("user", "unknown")),
                content=self._substitute_mentions(msg.get("text", ""), user_display_names),
                ts=msg.get("ts"),
                participant_name=user_display_names.get(participant_id, "unknown"),
            )
            for msg in messages
        ]
        # Decided from the untrimmed history, before _apply_context_scope
        # shrinks it down to 0-1 items regardless of the thread's real length.
        is_first_message = len(thread_history) == 1
        return thread_history, is_first_message

    @staticmethod
    def _apply_context_scope(
        thread_history: List[ThreadMessage], scope: str, trigger_ts: Optional[str]
    ) -> List[ThreadMessage]:
        """Trims thread_history down to the configured scope. The triggering
        message itself is excluded from the trimmed scopes since it's already
        sent separately as the query."""
        if scope == "full_thread":
            return thread_history

        without_trigger = [m for m in thread_history if m.ts != trigger_ts]
        if scope == "parent_message":
            return without_trigger[:1]
        if scope == "last_message":
            return without_trigger[-1:]
        return thread_history

    _SUMMARY_INSTRUCTION = (
        "Summarize this Slack conversation for a support agent picking it up. "
        "Preserve concrete facts: names, IDs, platforms/SDKs mentioned, error "
        "messages, and any information the user has already provided, so the "
        "agent doesn't need to ask for it again. Keep it concise."
    )

    def _summarize_thread_history(
        self, thread_history: List[ThreadMessage], model_config: Optional[Mapping] = None
    ) -> List[ThreadMessage]:
        """Condenses thread_history into a single AI-generated summary message.

        With no `model_config` (the "Summarization Model" setting left unset),
        uses Dify's built-in system summarization model - short transcripts
        are returned unchanged by the SDK itself (no LLM call), so this is
        safe to call unconditionally. When a specific model is configured,
        calls it directly instead, since the system-summary API has no way to
        pick a model.
        """
        if not thread_history:
            return thread_history

        transcript = "\n".join(
            f"{m.participant_name or m.participant_id}: {m.content}" for m in thread_history
        )
        try:
            if model_config:
                result = self.session.model.llm.invoke(
                    model_config=dict(model_config),
                    prompt_messages=[
                        SystemPromptMessage(content=self._SUMMARY_INSTRUCTION),
                        UserPromptMessage(content=transcript),
                    ],
                    stream=False,
                )
                summary = result.message.content
                if isinstance(summary, list):
                    summary = "".join(getattr(part, "data", "") or "" for part in summary)
            else:
                summary = self.session.model.summary.invoke(
                    text=transcript, instruction=self._SUMMARY_INSTRUCTION
                )
        except Exception as e:
            logger.warning("Thread history summarization failed, sending raw history instead: %s", e)
            return thread_history

        return [ThreadMessage(role="user", participant_id="summary", content=summary, participant_name="Thread Summary")]

    @staticmethod
    def _substitute_mentions(text: str, user_display_names: Mapping[str, str]) -> str:
        def replace(match: re.Match) -> str:
            user_id = match.group(1)
            return f"@{user_display_names[user_id]}" if user_id in user_display_names else match.group(0)

        # Slack HTML-escapes &, < and > in message text; undo that (after
        # mention substitution, which relies on the literal <@ID> form) so
        # the app sees what the user actually typed.
        text = MENTION_PATTERN.sub(replace, text or "")
        return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

    def _upload_slack_files(
        self, client: WebClient, ctx: RoutingContext, token: str, slack_files: List[Dict[str, Any]]
    ) -> List[Any]:
        uploaded = []
        for f in slack_files:
            file_name = f.get("name")
            file_url = f.get("url_private_download")
            mimetype = f.get("mimetype", "application/octet-stream")
            if not file_url or not file_name:
                continue

            try:
                resp = requests.get(file_url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
            except requests.RequestException as e:
                logger.warning("Failed to download file from Slack: %s (%s)", file_name, e)
                continue
            if resp.status_code != 200:
                logger.warning(
                    "Failed to download file from Slack: %s, status code=%s", file_name, resp.status_code
                )
                continue

            try:
                storage_file = self.session.file.upload(filename=file_name, content=resp.content, mimetype=mimetype)
                if storage_file:
                    uploaded.append(storage_file)
            except Exception as e:
                self._safe_post(
                    client,
                    ctx,
                    f"Error uploading file: {e}\n\n"
                    "This may be caused by an unconfigured `FILES_URL` in your `dify/docker/.env` .\n"
                    "Please set `FILES_URL` properly and restart( `docker compose down && docker compose up -d` ) "
                    "your Dify environment, then try again.",
                )
                logger.warning("Error uploading file via session.file.upload: %s", e)
        return uploaded

    @staticmethod
    def _build_invoke_inputs(
        ctx: RoutingContext,
        thread_history: List[ThreadMessage],
        user_display_names: Dict[str, str],
        uploaded_files: List[Any],
    ) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {
            "thread_history": json.dumps([m.to_dict() for m in thread_history], indent=4, ensure_ascii=False),
            "thread_users": json.dumps(user_display_names, indent=4, ensure_ascii=False),
            "thread_ts": ctx.slack_thread_ts,
            "channel_id": ctx.channel,
        }
        if uploaded_files:
            inputs["files"] = [
                {"type": f.type, "transfer_method": "remote_url", "url": f.preview_url}
                for f in uploaded_files
            ]
        return inputs

    def _handle_invoke_error(
        self, client: WebClient, ctx: RoutingContext, settings: Mapping, error: Exception
    ) -> Response:
        err_msg = str(error)
        err_trace = traceback.format_exc()

        skip_timeout_error = settings.get("skip_timeout_error", False)
        if skip_timeout_error and "invocation exited without response" in err_msg.lower():
            return Response(status=200, response="ok", content_type="text/plain")

        self._safe_post(
            client, ctx, f"Sorry, I'm having trouble processing your request. Please try again later. Error: {err_msg[:300]}"
        )
        return Response(
            status=200, response=f"An error occurred: {err_msg}\n{err_trace}", content_type="text/plain"
        )

    # ------------------------------------------------------------- replying

    def _post_answer(
        self, client: WebClient, ctx: RoutingContext, answer: str, settings: Mapping, is_first_message: bool
    ) -> Response:
        converted_answer = SlackMarkdownConverter().convert(answer)
        chunks = self._split_into_chunks(converted_answer, SLACK_BLOCK_TEXT_LIMIT)
        # reply_broadcast only makes sense on a threaded reply, never a plain DM message.
        reply_broadcast = bool(
            settings.get("first_reply_broadcast", False)
            and is_first_message
            and ctx.reply_thread_ts
            and not ctx.is_dm
        )

        for i, chunk in enumerate(chunks):
            resp = client.chat_postMessage(
                channel=ctx.channel,
                text=chunk,  # fallback text
                thread_ts=ctx.reply_thread_ts,
                blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": chunk}}],
                # Only broadcast the first chunk - broadcasting every chunk
                # would flood the channel outside the thread.
                reply_broadcast=reply_broadcast if i == 0 else False,
            )
            self._append_thread_message(
                ctx.channel,
                ctx.conversation_ts,
                {
                    "ts": resp.get("ts"),
                    "text": chunk,
                    "user": resp.get("message", {}).get("user"),
                    "bot_id": resp.get("message", {}).get("bot_id"),
                },
            )

        return Response(status=200, response="ok", content_type="text/plain")

    @staticmethod
    def _split_into_chunks(text: str, max_len: int) -> List[str]:
        """Greedily packs lines into chunks up to `max_len` characters,
        splitting any single line that's longer than max_len on its own."""
        if len(text) <= max_len:
            return [text]

        def pieces(line: str):
            if len(line) <= max_len:
                yield line
            else:
                for i in range(0, len(line), max_len):
                    yield line[i : i + max_len]

        chunks: List[str] = []
        current = ""
        for line in text.split("\n"):
            for piece in pieces(line):
                added_len = len(piece) + (1 if current else 0)
                if len(current) + added_len <= max_len:
                    current = f"{current}\n{piece}" if current else piece
                else:
                    if current:
                        chunks.append(current)
                    current = piece
        if current:
            chunks.append(current)
        return chunks
