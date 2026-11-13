"""Hartwell Foods HR email agent: drafts replies to employee emails using Claude's tool use.

Two modes:

  --email FILE   Handle one .eml file. save_reply writes the draft to week-04/drafts/.
  --inbox        Sign in to the Gmail sandbox with IMAP (same login as load_inbox.py, read from
                 Agent-Email.txt in the folder above the repository) and handle every unread
                 email in the inbox, one at a time, each as its own separate agent run. Each
                 email is marked as read *before* work starts, so no email is handled twice.
                 save_reply creates a real Gmail draft replying to the sender in the same thread;
                 if needs_human_review is true, the original email and the draft get the Gmail
                 label "Needs HR Review" (Gmail creates the label the first time).

For each email, the agent:
  1. reads the sender's address from the email's From header,
  2. starts hr_mcp_server.py --as-employee <sender> and offers Claude that server's tools
     (search_policy, get_leave_balance), plus its own save_reply tool,
  3. loops until Claude stops asking for tools, or 10 tool calls have run,
  4. prints each step, the saved draft, and the total input and output tokens.

This program only reads email and creates drafts. It has no way to send email.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-04\\agent.py --email week-04\\hartwell-hr\\sample-emails\\01-dana-vacation.eml
    .venv\\Scripts\\python.exe week-04\\agent.py --inbox
    .venv\\Scripts\\python.exe week-04\\agent.py --inbox --save week-04\\runs\\inbox-run.txt

--save FILE also writes everything this program prints to FILE (UTF-8). Log lines from the
hr_mcp_server.py subprocess still appear on screen but are not written to the file.

API key: read from ANTHROPIC_API_KEY if set, otherwise from the key file in the folder above the
repository. The key and the email password are only held in memory; they are never printed or
written anywhere.
"""

import argparse
import asyncio
import email
import email.policy
import imaplib
import json
import os
import re
import sys
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr
from pathlib import Path

import anthropic
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from load_inbox import CREDS_FILE, IMAP_HOST, load_login

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SERVER = HERE / "hr_mcp_server.py"
SYSTEM_PROMPT_FILE = HERE / "hartwell-hr" / "system_prompt.txt"
DRAFTS_DIR = HERE / "drafts"
KEY_FILES = [REPO.parent / "API-Key.txt", REPO.parent / "API Key.txt"]

MODEL = "claude-sonnet-5"
MAX_TOOL_CALLS = 10
PREVIEW_CHARS = 300
REVIEW_LABEL = "Needs HR Review"  # Gmail label for conversations that need a person from HR

SAVE_REPLY_TOOL = {
    "name": "save_reply",
    "description": (
        "Save the finished reply to the employee as a draft for the HR team to review. "
        "This only saves a draft; it never sends anything. Call it exactly once, at the end."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reply_text": {
                "type": "string",
                "description": "The full text of the reply email to the employee, greeting through sign-off.",
            },
            "needs_human_review": {
                "type": "boolean",
                "description": (
                    "true if a person from HR must handle or check this before anything is sent "
                    "(sensitive topics, or the policies don't answer the question); otherwise false."
                ),
            },
        },
        "required": ["reply_text", "needs_human_review"],
        "additionalProperties": False,
    },
    "strict": True,
}


# ---------- setup ----------

class Tee:
    """Write everything to the console and to a log file."""

    def __init__(self, stream, file):
        self._stream, self._file = stream, file

    def write(self, s):
        self._stream.write(s)
        self._file.write(s)
        return len(s)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def __getattr__(self, name):  # fileno, isatty, encoding, ... come from the console stream
        return getattr(self._stream, name)


def load_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    for path in KEY_FILES:
        if path.is_file():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key
    sys.exit("No API key found. Set ANTHROPIC_API_KEY or put the key in "
             f"{KEY_FILES[0].name} in {REPO.parent}.")


def parse_email(msg):
    """Return (sender_address, subject, full_text) for a parsed email. Raises ValueError."""
    _, sender = parseaddr(str(msg.get("From", "")))
    sender = sender.strip()
    if not sender or "@" not in sender:
        raise ValueError("Could not read a sender address from the From header.")
    subject = str(msg.get("Subject", "")).strip()
    body_part = msg.get_body(preferencelist=("plain", "html"))
    body = body_part.get_content().strip() if body_part else ""
    text = f"From: {msg.get('From', '')}\nSubject: {subject}\n\n{body}"
    return sender, subject, text


def reply_subject(subject):
    return subject if subject.lower().startswith("re:") else f"Re: {subject}"


def mcp_tool_to_claude(tool):
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.input_schema,
    }


def mcp_result_text(result):
    """Flatten an MCP CallToolResult into text for Claude."""
    parts = [c.text for c in (getattr(result, "content", None) or []) if getattr(c, "text", None)]
    if parts:
        return "\n".join(parts)
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return json.dumps(structured, indent=2)
    return "(empty result)"


# ---------- save_reply: file drafts (--email) ----------

def save_reply_file(email_path, sender, subject, reply_text, needs_human_review):
    """Write the reply as a draft file. Saves only; never sends."""
    DRAFTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(email_path).stem)
    path = DRAFTS_DIR / f"{stamp}-{stem}.txt"
    draft = (
        f"To: {sender}\n"
        f"Subject: {reply_subject(subject)}\n"
        f"Needs human review: {'YES' if needs_human_review else 'no'}\n"
        f"Status: DRAFT - not sent\n"
        f"\n{reply_text.strip()}\n"
    )
    path.write_text(draft, encoding="utf-8")
    where = str(path.relative_to(REPO))
    return f"Draft saved to {where}. It has not been sent.", where, draft


# ---------- Gmail over IMAP (--inbox): read, mark read, create drafts. No sending. ----------

class GmailSandbox:
    """Minimal IMAP client for the sandbox mailbox. It can read mail and add drafts only."""

    def __init__(self):
        self.user, self._password = load_login(CREDS_FILE)  # password stays in memory only
        self.imap = None
        self.drafts_box = None

    def connect(self):
        try:
            imap = imaplib.IMAP4_SSL(IMAP_HOST)
            imap.login(self.user, self._password)
        except imaplib.IMAP4.error:
            # Never echo the server reply or anything that could include the password.
            sys.exit("IMAP login failed. Check the address/app password in Agent-Email.txt "
                     "and that IMAP is enabled for the account.")
        self.imap = imap
        self._check(self.imap.select("INBOX"), "open INBOX")
        if self.drafts_box is None:
            self.drafts_box = self._find_drafts_mailbox()

    def close(self):
        if self.imap is not None:
            try:
                self.imap.logout()
            except Exception:
                pass
            self.imap = None

    def _call(self, op):
        """Run op(imap); if the connection dropped (e.g. during a long agent run), reconnect once."""
        try:
            return op(self.imap)
        except (imaplib.IMAP4.abort, OSError):
            self.close()
            self.connect()
            return op(self.imap)

    @staticmethod
    def _check(result, what):
        status, data = result
        if status != "OK":
            raise RuntimeError(f"IMAP could not {what}: {data!r}")
        return data

    def _find_drafts_mailbox(self):
        """Gmail marks its drafts folder with the \\Drafts attribute; its name varies by language."""
        for line in self._check(self.imap.list(), "list mailboxes"):
            text = line.decode("utf-8", "replace") if isinstance(line, bytes) else str(line)
            if "\\Drafts" in text:
                m = re.search(r'"([^"]+)"\s*$', text) or re.search(r"(\S+)\s*$", text)
                if m:
                    return m.group(1)
        return "[Gmail]/Drafts"

    def unread_uids(self):
        data = self._check(self._call(lambda i: i.uid("SEARCH", None, "UNSEEN")), "search INBOX")
        return data[0].split() if data and data[0] else []

    def claim(self, uid):
        """Mark one message as read, then fetch it. Returns raw bytes, or None if it's gone."""
        self._check(self._call(lambda i: i.uid("STORE", uid, "+FLAGS", r"(\Seen)")), "mark as read")
        data = self._check(self._call(lambda i: i.uid("FETCH", uid, "(BODY.PEEK[])")), "fetch message")
        for part in data:
            if isinstance(part, tuple) and len(part) == 2:
                return part[1]
        return None

    def add_label(self, mailbox, uid, label):
        """Put a Gmail label on one message (Gmail creates the label if it doesn't exist yet)."""
        def op(i):
            self._check(i.select(f'"{mailbox}"'), f"open {mailbox}")
            try:
                return i.uid("STORE", uid, "+X-GM-LABELS", f'("{label}")')
            finally:
                i.select("INBOX")
        self._check(self._call(op), f"add the label {label!r}")

    def create_draft(self, original, original_uid, to_addr, subject, reply_text, needs_human_review):
        """Add a reply draft to Gmail's Drafts folder, threaded to the original. Never sends.

        If needs_human_review, both the original email and the draft get the REVIEW_LABEL label.
        """
        msg = EmailMessage()
        msg["From"] = self.user
        msg["To"] = to_addr
        msg["Subject"] = reply_subject(subject)
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid()
        orig_id = str(original.get("Message-ID", "")).strip()
        if orig_id:
            refs = str(original.get("References", "")).split()
            msg["In-Reply-To"] = orig_id
            msg["References"] = " ".join(refs + [orig_id])
        msg.set_content(reply_text.strip() + "\n")

        raw = msg.as_bytes(policy=email.policy.SMTP)
        box = f'"{self.drafts_box}"'
        resp = self._check(
            self._call(lambda i: i.append(box, r"(\Draft \Seen)",
                                          imaplib.Time2Internaldate(datetime.now().timestamp()), raw)),
            "create the draft",
        )

        label_note = "no"
        if needs_human_review:
            # Label the original in INBOX (shows in the inbox list) and the new draft (shows in Drafts).
            self.add_label("INBOX", original_uid, REVIEW_LABEL)
            m = re.search(rb"APPENDUID \d+ (\d+)", b" ".join(r for r in resp if isinstance(r, bytes)))
            if m:
                self.add_label(self.drafts_box, m.group(1).decode(), REVIEW_LABEL)
            label_note = f"YES - Gmail label '{REVIEW_LABEL}' added to the conversation"

        draft = (
            f"To: {to_addr}\n"
            f"Subject: {msg['Subject']}\n"
            f"In-Reply-To: {orig_id or '(original had no Message-ID; not threaded)'}\n"
            f"Needs human review: {label_note}\n"
            f"Status: GMAIL DRAFT - not sent\n"
            f"\n{reply_text.strip()}\n"
        )
        return draft


# ---------- agent loop ----------

def show_step(n, name, inputs, result_text, is_error):
    print(f"\n--- Step {n}: {name}{'  [ERROR]' if is_error else ''}")
    print(f"Inputs: {json.dumps(inputs, ensure_ascii=False)}")
    preview = result_text[:PREVIEW_CHARS]
    more = "..." if len(result_text) > PREVIEW_CHARS else ""
    print(f"Result: {preview}{more}")


async def run_agent(client, system_prompt, sender, email_text, save_fn):
    """One agent run for one email. save_fn(reply_text, needs_review) -> (tool_text, where, draft).

    Returns a dict with tool_calls, input_tokens, output_tokens and saved (True/False).
    The MCP client runs in task groups, which wrap any error in an ExceptionGroup; unwrap a
    single error so callers can catch e.g. anthropic.AuthenticationError directly.
    """
    try:
        return await _run_agent(client, system_prompt, sender, email_text, save_fn)
    except BaseExceptionGroup as group:
        err = group
        while isinstance(err, BaseExceptionGroup) and len(err.exceptions) == 1:
            err = err.exceptions[0]
        if err is group:
            raise
        raise err from None


async def _run_agent(client, system_prompt, sender, email_text, save_fn):
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER), "--as-employee", sender],
        cwd=str(HERE),
    )

    tool_calls = 0
    input_tokens = output_tokens = 0
    saved = None  # (where, draft) of the last save_reply
    stop_reason_text = None

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            mcp_names = {t.name for t in mcp_tools}
            tools = [mcp_tool_to_claude(t) for t in mcp_tools] + [SAVE_REPLY_TOOL]
            print(f"Tools:  {', '.join(t['name'] for t in tools)}")

            messages = [{
                "role": "user",
                "content": (
                    "Prepare a reply to this employee email.\n\n"
                    f"<email>\n{email_text}\n</email>"
                ),
            }]

            while True:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=16000,
                    system=system_prompt,
                    tools=tools,
                    messages=messages,
                )
                input_tokens += response.usage.input_tokens
                output_tokens += response.usage.output_tokens
                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason != "tool_use":
                    reasons = {
                        "end_turn": "Claude finished and asked for no more tools.",
                        "max_tokens": "Claude's reply hit the max_tokens limit.",
                        "refusal": "Claude declined to continue (refusal).",
                    }
                    stop_reason_text = reasons.get(
                        response.stop_reason,
                        f"Claude's reply was not a tool request (stop_reason: {response.stop_reason}).",
                    )
                    final_text = "".join(b.text for b in response.content if b.type == "text").strip()
                    if final_text:
                        print(f"\nClaude: {final_text}")
                    break

                results = []
                limit_hit = False
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    if tool_calls >= MAX_TOOL_CALLS:
                        limit_hit = True
                        break
                    tool_calls += 1
                    is_error = False
                    try:
                        if block.name == "save_reply":
                            text, where, draft = save_fn(
                                block.input["reply_text"], bool(block.input["needs_human_review"]),
                            )
                            saved = (where, draft)
                        elif block.name in mcp_names:
                            result = await session.call_tool(block.name, block.input or {})
                            text = mcp_result_text(result)
                            is_error = bool(getattr(result, "is_error", False))
                        else:
                            text, is_error = f"Unknown tool: {block.name}", True
                    except Exception as e:  # report tool failures back to Claude instead of crashing
                        text, is_error = f"Tool failed: {e}", True
                    show_step(tool_calls, block.name, block.input, text, is_error)
                    results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": text, "is_error": is_error,
                    })

                if limit_hit or tool_calls >= MAX_TOOL_CALLS:
                    stop_reason_text = f"Stopped: reached the limit of {MAX_TOOL_CALLS} tool calls."
                    break
                messages.append({"role": "user", "content": results})

    print(f"\n=== Stopped: {stop_reason_text}")
    if saved:
        where, draft = saved
        print(f"\n=== Draft ({where}) ===\n{draft}")
    else:
        print("\n=== No draft was saved. ===")
    print(f"Tool calls: {tool_calls}")
    print(f"Tokens: {input_tokens:,} input, {output_tokens:,} output")
    return {"tool_calls": tool_calls, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "saved": saved is not None}


# ---------- modes ----------

async def run_file(email_path):
    client = anthropic.Anthropic(api_key=load_api_key())
    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    with open(email_path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=email.policy.default)
    try:
        sender, subject, email_text = parse_email(msg)
    except ValueError as e:
        sys.exit(f"{e} ({email_path})")

    print(f"Email:  {email_path}")
    print(f"Sender: {sender}")
    print(f"Model:  {MODEL}")

    def save_fn(reply_text, needs_review):
        return save_reply_file(email_path, sender, subject, reply_text, needs_review)

    await run_agent(client, system_prompt, sender, email_text, save_fn)


async def run_inbox():
    client = anthropic.Anthropic(api_key=load_api_key())
    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    # Free check that the key and model work *before* any email is marked as read.
    client.models.retrieve(MODEL)
    gmail = GmailSandbox()
    # Don't print the login address: it comes from Agent-Email.txt and --save writes output to a file.
    print(f"Signing in to {IMAP_HOST} with the login from Agent-Email.txt ...")
    gmail.connect()
    print(f"Drafts folder: {gmail.drafts_box}")
    print(f"Model: {MODEL}")

    totals = {"emails": 0, "drafts": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0}
    try:
        uids = gmail.unread_uids()
        print(f"Unread emails in INBOX: {len(uids)}")
        for n, uid in enumerate(uids, start=1):
            print(f"\n{'=' * 70}\nEmail {n} of {len(uids)}  (IMAP UID {uid.decode()})")
            raw = gmail.claim(uid)  # marked as read before any work starts
            if raw is None:
                print("Message is no longer in the inbox; skipped.")
                continue
            totals["emails"] += 1
            msg = email.message_from_bytes(raw, policy=email.policy.default)
            try:
                sender, subject, email_text = parse_email(msg)
            except ValueError as e:
                totals["failed"] += 1
                print(f"Skipped: {e} (it is now marked as read).")
                continue
            print(f"Sender:  {sender}")
            print(f"Subject: {subject}")

            def save_fn(reply_text, needs_review, msg=msg, uid=uid, sender=sender, subject=subject):
                draft = gmail.create_draft(msg, uid, sender, subject, reply_text, needs_review)
                note = f" The conversation is labeled '{REVIEW_LABEL}'." if needs_review else ""
                return ("Draft created in the Gmail Drafts folder, threaded as a reply. "
                        f"It has not been sent.{note}"), f"Gmail {gmail.drafts_box}", draft

            try:
                stats = await run_agent(client, system_prompt, sender, email_text, save_fn)
            except anthropic.AuthenticationError:
                raise  # a bad API key would fail every email; stop now
            except Exception as e:
                totals["failed"] += 1
                print(f"\n!!! This email could not be handled ({type(e).__name__}: {e}). "
                      "It is marked as read; mark it unread in Gmail to retry it.")
                continue
            totals["drafts"] += stats["saved"]
            totals["input_tokens"] += stats["input_tokens"]
            totals["output_tokens"] += stats["output_tokens"]
    finally:
        gmail.close()

    print(f"\n{'=' * 70}\nInbox run finished.")
    print(f"Emails handled: {totals['emails']}  |  Gmail drafts created: {totals['drafts']}  |  "
          f"Failed/skipped: {totals['failed']}")
    print(f"Total tokens: {totals['input_tokens']:,} input, {totals['output_tokens']:,} output")


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Draft replies to HR emails with Claude. Never sends email.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--email", help="path of one email file (.eml); the draft is saved to week-04/drafts/")
    mode.add_argument("--inbox", action="store_true",
                      help="handle every unread email in the Gmail sandbox inbox; drafts go to Gmail Drafts")
    parser.add_argument("--save", type=Path, metavar="FILE",
                        help="also write everything printed to FILE (UTF-8)")
    args = parser.parse_args()
    if args.email and not Path(args.email).is_file():
        sys.exit(f"Email file not found: {args.email}")

    log_file = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.save, "w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)

    error = None
    try:
        asyncio.run(run_inbox() if args.inbox else run_file(args.email))
    except anthropic.AuthenticationError:
        error = "The Anthropic API rejected the API key."
    except anthropic.RateLimitError:
        error = "Rate limited by the Anthropic API; try again shortly."
    except anthropic.APIStatusError as e:
        error = f"Anthropic API error {e.status_code}: {e.message}"
    except anthropic.APIConnectionError:
        error = "Could not reach the Anthropic API."
    except SystemExit as e:  # e.g. IMAP login failed, bad email file
        if not isinstance(e.code, str):
            raise
        error = e.code
    finally:
        if error:
            print(error, file=sys.stderr)  # goes to the --save file too, while Tee is active
        if log_file is not None:
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            log_file.close()
            print(f"Output also saved to {args.save}")
    if error:
        sys.exit(1)


if __name__ == "__main__":
    main()
