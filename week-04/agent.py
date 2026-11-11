"""Hartwell Foods HR email agent: drafts a reply to one employee email using Claude's tool use.

For the email given with --email, the agent:
  1. reads the sender's address from the email's From header,
  2. starts hr_mcp_server.py --as-employee <sender> and offers Claude that server's tools
     (search_policy, get_leave_balance), plus its own save_reply tool,
  3. loops until Claude stops asking for tools, or 10 tool calls have run,
  4. prints each step, the saved draft, and the total input and output tokens.

save_reply only writes a draft to week-04/drafts/. Nothing is ever sent.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-04\\agent.py --email week-04\\hartwell-hr\\sample-emails\\01-dana-vacation.eml

API key: read from ANTHROPIC_API_KEY if set, otherwise from the key file in the folder above the
repository. The key is only held in memory; it is never written anywhere.
"""

import argparse
import asyncio
import email
import email.policy
import json
import os
import re
import sys
from datetime import datetime
from email.utils import parseaddr
from pathlib import Path

import anthropic
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SERVER = HERE / "hr_mcp_server.py"
SYSTEM_PROMPT_FILE = HERE / "hartwell-hr" / "system_prompt.txt"
DRAFTS_DIR = HERE / "drafts"
KEY_FILES = [REPO.parent / "API-Key.txt", REPO.parent / "API Key.txt"]

MODEL = "claude-sonnet-5"
MAX_TOOL_CALLS = 10
PREVIEW_CHARS = 300

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


def read_email(path):
    """Return (sender_address, subject, full_text) for one .eml file."""
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=email.policy.default)
    _, sender = parseaddr(str(msg.get("From", "")))
    sender = sender.strip()
    if not sender or "@" not in sender:
        sys.exit(f"Could not read a sender address from the From header of {path}.")
    subject = str(msg.get("Subject", "")).strip()
    body_part = msg.get_body(preferencelist=("plain", "html"))
    body = body_part.get_content().strip() if body_part else ""
    text = f"From: {msg.get('From', '')}\nSubject: {subject}\n\n{body}"
    return sender, subject, text


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


# ---------- save_reply ----------

def save_reply(email_path, sender, subject, reply_text, needs_human_review):
    """Write the reply as a draft file. Saves only; never sends."""
    DRAFTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(email_path).stem)
    path = DRAFTS_DIR / f"{stamp}-{stem}.txt"
    re_subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    draft = (
        f"To: {sender}\n"
        f"Subject: {re_subject}\n"
        f"Needs human review: {'YES' if needs_human_review else 'no'}\n"
        f"Status: DRAFT - not sent\n"
        f"\n{reply_text.strip()}\n"
    )
    path.write_text(draft, encoding="utf-8")
    return path, draft


# ---------- agent loop ----------

def show_step(n, name, inputs, result_text, is_error):
    print(f"\n--- Step {n}: {name}{'  [ERROR]' if is_error else ''}")
    print(f"Inputs: {json.dumps(inputs, ensure_ascii=False)}")
    preview = result_text[:PREVIEW_CHARS]
    more = "..." if len(result_text) > PREVIEW_CHARS else ""
    print(f"Result: {preview}{more}")


async def run(email_path):
    api_key = load_api_key()
    sender, subject, email_text = read_email(email_path)
    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    client = anthropic.Anthropic(api_key=api_key)

    print(f"Email:  {email_path}")
    print(f"Sender: {sender}")
    print(f"Model:  {MODEL}")

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER), "--as-employee", sender],
        cwd=str(HERE),
    )

    tool_calls = 0
    input_tokens = output_tokens = 0
    saved = None  # (path, draft) of the last save_reply
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
                            path, draft = save_reply(
                                email_path, sender, subject,
                                block.input["reply_text"], bool(block.input["needs_human_review"]),
                            )
                            saved = (path, draft)
                            text = f"Draft saved to {path.relative_to(REPO)}. It has not been sent."
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
        path, draft = saved
        print(f"\n=== Draft ({path.relative_to(REPO)}) ===\n{draft}")
    else:
        print("\n=== No draft was saved. ===")
    print(f"Tool calls: {tool_calls}")
    print(f"Tokens: {input_tokens:,} input, {output_tokens:,} output")


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Draft a reply to one HR email with Claude.")
    parser.add_argument("--email", required=True, help="path of one email file (.eml)")
    args = parser.parse_args()
    if not Path(args.email).is_file():
        sys.exit(f"Email file not found: {args.email}")
    try:
        asyncio.run(run(args.email))
    except anthropic.AuthenticationError:
        sys.exit("The Anthropic API rejected the API key.")
    except anthropic.RateLimitError:
        sys.exit("Rate limited by the Anthropic API; try again shortly.")
    except anthropic.APIStatusError as e:
        sys.exit(f"Anthropic API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        sys.exit("Could not reach the Anthropic API.")


if __name__ == "__main__":
    main()
