"""Load the Hartwell HR sample emails into a Gmail sandbox inbox over IMAP.

Each .eml file in hartwell-hr/sample-emails is appended straight into INBOX as an
unread message dated today. Nothing is sent over SMTP.

The login is read at run time from Agent-Email.txt in the folder above the repo
(never stored in this repo). The password is never printed or saved.

Agent-Email.txt can be either:
    email: someone@gmail.com          (or user= / login: / address: ...)
    password: abcd efgh ijkl mnop     (a Gmail app password)
or just two lines: the email address, then the password.

Usage (from the repo root):
    .venv\\Scripts\\python.exe week-04\\load_inbox.py
"""

import imaplib
import re
import sys
import time
from email import policy
from email.parser import BytesParser
from email.utils import formatdate, make_msgid
from pathlib import Path

HERE = Path(__file__).resolve().parent
EMAIL_DIR = HERE / "hartwell-hr" / "sample-emails"
CREDS_FILE = HERE.parent.parent / "Agent-Email.txt"  # folder above the repo
IMAP_HOST = "imap.gmail.com"
MAILBOX = "INBOX"


def load_login(path: Path) -> tuple[str, str]:
    """Return (email, password) from the credentials file without echoing either."""
    if not path.is_file():
        sys.exit(f"Credentials file not found: {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]

    user = password = None
    for ln in lines:
        m = re.match(r"^\s*([A-Za-z _-]+?)\s*[:=]\s*(.+?)\s*$", ln)
        if not m:
            continue
        key, value = m.group(1).lower(), m.group(2).strip().strip("\"'")
        if "pass" in key:
            password = value
        elif any(k in key for k in ("email", "user", "login", "address", "account")):
            user = value

    if user is None or password is None:
        # Fallback: first line is the address, second is the password.
        if len(lines) >= 2 and "@" in lines[0]:
            user, password = lines[0], lines[1]
        else:
            sys.exit("Could not find an email address and password in Agent-Email.txt "
                     "(expected 'email: ...' and 'password: ...' lines).")

    # Gmail shows app passwords as four groups of four; IMAP wants them without spaces.
    if re.fullmatch(r"(?:[a-zA-Z]{4} ){3}[a-zA-Z]{4}", password):
        password = password.replace(" ", "")
    return user, password


def prepare_message(raw: bytes) -> bytes:
    """Stamp the message with today's date and a Message-ID."""
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    del msg["Date"]
    msg["Date"] = formatdate(localtime=True)
    if msg["Message-ID"] is None:
        msg["Message-ID"] = make_msgid(domain="hartwellfoods.example")
    return msg.as_bytes(policy=policy.SMTP)  # CRLF line endings for IMAP


def main() -> None:
    files = sorted(EMAIL_DIR.glob("*.eml"))
    if not files:
        sys.exit(f"No .eml files found in {EMAIL_DIR}")

    user, password = load_login(CREDS_FILE)
    print(f"Signing in to {IMAP_HOST} with the login from Agent-Email.txt ...")

    try:
        imap = imaplib.IMAP4_SSL(IMAP_HOST)
        imap.login(user, password)
    except imaplib.IMAP4.error:
        # Don't echo anything that could include the password.
        sys.exit("IMAP login failed. Check the address/app password in Agent-Email.txt "
                 "and that IMAP is enabled for the account.")
    finally:
        password = None  # drop our reference as soon as we're done with it

    try:
        now = imaplib.Time2Internaldate(time.time())
        for path in files:
            data = prepare_message(path.read_bytes())
            # Empty flags => no \Seen, so the message arrives unread.
            status, resp = imap.append(MAILBOX, "", now, data)
            if status != "OK":
                print(f"  FAILED  {path.name}: {resp}")
            else:
                print(f"  added   {path.name}")
    finally:
        imap.logout()

    print(f"Done. {len(files)} message(s) placed in {MAILBOX} as unread.")


if __name__ == "__main__":
    main()
