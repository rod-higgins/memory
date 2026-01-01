"""Communication data source adapters (Email, Slack, Discord, iMessage)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

from .base import DataCategory, DataPoint, DataSource


class iMessageSource(DataSource):
    """Adapter for Apple iMessage/Messages database."""

    source_type = "imessage"
    category = DataCategory.COMMUNICATION

    def __init__(
        self,
        db_path: str = "~/Library/Messages/chat.db",
        **kwargs,
    ):
        """
        Initialize iMessage source.

        Args:
            db_path: Path to Messages database (requires Full Disk Access)
        """
        super().__init__(**kwargs)
        self.db_path = Path(db_path).expanduser()

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over iMessage conversations."""
        if not self.db_path.exists():
            return

        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row

            # Query messages with contact info
            query = """
                SELECT
                    m.ROWID,
                    m.text,
                    m.date,
                    m.is_from_me,
                    h.id as handle_id,
                    c.display_name as chat_name
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.ROWID
                LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
                LEFT JOIN chat c ON cmj.chat_id = c.ROWID
                WHERE m.text IS NOT NULL AND m.text != ''
                ORDER BY m.date DESC
                LIMIT 10000
            """

            cursor = conn.execute(query)

            for row in cursor:
                # Convert Apple's timestamp (nanoseconds since 2001-01-01)
                if row["date"]:
                    timestamp = datetime(2001, 1, 1) + \
                        __import__("datetime").timedelta(seconds=row["date"] / 1_000_000_000)
                else:
                    timestamp = None

                direction = "sent" if row["is_from_me"] else "received"
                contact = row["handle_id"] or "Unknown"
                chat = row["chat_name"] or contact

                yield DataPoint(
                    content=row["text"],
                    category=self.category,
                    source_type=self.source_type,
                    source_identifier=f"message_{row['ROWID']}",
                    timestamp=timestamp,
                    metadata={
                        "direction": direction,
                        "contact": contact,
                        "chat": chat,
                        "is_from_me": bool(row["is_from_me"]),
                    },
                    tags=["message", direction],
                )

            conn.close()

        except sqlite3.OperationalError as e:
            # Permission denied or database locked
            raise PermissionError(
                f"Cannot access Messages database. Grant Full Disk Access. Error: {e}"
            )


class SlackExportSource(DataSource):
    """Adapter for Slack workspace export."""

    source_type = "slack"
    category = DataCategory.COMMUNICATION

    def __init__(
        self,
        export_path: str,
        include_channels: list[str] | None = None,
        include_dms: bool = True,
        **kwargs,
    ):
        """
        Initialize Slack export source.

        Args:
            export_path: Path to extracted Slack export
            include_channels: List of channel names to include (None = all)
            include_dms: Whether to include direct messages
        """
        super().__init__(**kwargs)
        self.export_path = Path(export_path).expanduser()
        self.include_channels = set(include_channels) if include_channels else None
        self.include_dms = include_dms

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over Slack messages."""
        if not self.export_path.exists():
            return

        # Load users for name resolution
        users = {}
        users_file = self.export_path / "users.json"
        if users_file.exists():
            with open(users_file) as f:
                for user in json.load(f):
                    users[user["id"]] = user.get("real_name") or user.get("name", "Unknown")

        # Load channels
        channels_file = self.export_path / "channels.json"
        channels = {}
        if channels_file.exists():
            with open(channels_file) as f:
                for ch in json.load(f):
                    channels[ch["id"]] = ch["name"]

        # Process each channel directory
        for channel_dir in self.export_path.iterdir():
            if not channel_dir.is_dir():
                continue

            channel_name = channel_dir.name

            # Filter channels
            if self.include_channels and channel_name not in self.include_channels:
                continue

            # Process message files (one per day)
            for msg_file in sorted(channel_dir.glob("*.json")):
                with open(msg_file) as f:
                    messages = json.load(f)

                for msg in messages:
                    if msg.get("type") != "message":
                        continue

                    text = msg.get("text", "")
                    if not text:
                        continue

                    # Parse timestamp
                    ts = msg.get("ts", "0")
                    try:
                        timestamp = datetime.fromtimestamp(float(ts))
                    except (ValueError, OSError):
                        timestamp = None

                    user_id = msg.get("user", "")
                    user_name = users.get(user_id, user_id)

                    yield DataPoint(
                        content=text,
                        category=self.category,
                        source_type=self.source_type,
                        source_identifier=f"slack_{channel_name}_{ts}",
                        timestamp=timestamp,
                        metadata={
                            "channel": channel_name,
                            "user_id": user_id,
                            "user_name": user_name,
                            "thread_ts": msg.get("thread_ts"),
                            "reactions": [r["name"] for r in msg.get("reactions", [])],
                        },
                        tags=["slack", channel_name],
                    )


class DiscordExportSource(DataSource):
    """Adapter for Discord data export."""

    source_type = "discord"
    category = DataCategory.COMMUNICATION

    def __init__(self, export_path: str, **kwargs):
        """
        Initialize Discord export source.

        Args:
            export_path: Path to Discord data package
        """
        super().__init__(**kwargs)
        self.export_path = Path(export_path).expanduser()

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over Discord messages."""
        messages_dir = self.export_path / "messages"
        if not messages_dir.exists():
            return

        for channel_dir in messages_dir.iterdir():
            if not channel_dir.is_dir():
                continue

            messages_file = channel_dir / "messages.json"
            if not messages_file.exists():
                continue

            # Load channel info
            channel_info = {}
            channel_json = channel_dir / "channel.json"
            if channel_json.exists():
                with open(channel_json) as f:
                    channel_info = json.load(f)

            channel_name = channel_info.get("name", channel_dir.name)
            channel_type = channel_info.get("type", "unknown")

            with open(messages_file) as f:
                messages = json.load(f)

            for msg in messages:
                content = msg.get("Contents", "")
                if not content:
                    continue

                # Parse timestamp
                timestamp = None
                ts_str = msg.get("Timestamp")
                if ts_str:
                    try:
                        timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except ValueError:
                        pass

                yield DataPoint(
                    content=content,
                    category=self.category,
                    source_type=self.source_type,
                    source_identifier=f"discord_{msg.get('ID', '')}",
                    timestamp=timestamp,
                    metadata={
                        "channel": channel_name,
                        "channel_type": channel_type,
                        "attachments": msg.get("Attachments", ""),
                    },
                    tags=["discord", channel_name],
                )


class WhatsAppExportSource(DataSource):
    """Adapter for WhatsApp chat export."""

    source_type = "whatsapp"
    category = DataCategory.COMMUNICATION

    def __init__(self, export_path: str, **kwargs):
        """
        Initialize WhatsApp export source.

        Args:
            export_path: Path to WhatsApp chat export (txt file or folder)
        """
        super().__init__(**kwargs)
        self.export_path = Path(export_path).expanduser()

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over WhatsApp messages."""
        import re

        # WhatsApp export format: [DD/MM/YYYY, HH:MM:SS] Sender: Message
        # or: DD/MM/YYYY, HH:MM - Sender: Message
        patterns = [
            re.compile(r"\[(\d{1,2}/\d{1,2}/\d{4}),\s+(\d{1,2}:\d{2}:\d{2})\]\s+([^:]+):\s+(.+)"),
            re.compile(r"(\d{1,2}/\d{1,2}/\d{4}),\s+(\d{1,2}:\d{2})\s+-\s+([^:]+):\s+(.+)"),
        ]

        files_to_process = []
        if self.export_path.is_file():
            files_to_process = [self.export_path]
        elif self.export_path.is_dir():
            files_to_process = list(self.export_path.glob("*.txt"))

        for file_path in files_to_process:
            chat_name = file_path.stem

            with open(file_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    for pattern in patterns:
                        match = pattern.match(line)
                        if match:
                            date_str, time_str, sender, message = match.groups()

                            # Parse timestamp
                            timestamp = None
                            try:
                                # Try different date formats
                                for fmt in ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M"]:
                                    try:
                                        timestamp = datetime.strptime(
                                            f"{date_str} {time_str}", fmt
                                        )
                                        break
                                    except ValueError:
                                        continue
                            except Exception:
                                pass

                            yield DataPoint(
                                content=message,
                                category=self.category,
                                source_type=self.source_type,
                                source_identifier=f"whatsapp_{chat_name}_{line_num}",
                                timestamp=timestamp,
                                metadata={
                                    "chat": chat_name,
                                    "sender": sender.strip(),
                                },
                                tags=["whatsapp", chat_name],
                            )
                            break


class GmailExportSource(DataSource):
    """Adapter for Gmail/Google Takeout export."""

    source_type = "gmail"
    category = DataCategory.COMMUNICATION

    def __init__(self, export_path: str, **kwargs):
        """
        Initialize Gmail export source.

        Args:
            export_path: Path to Gmail Takeout export (mbox file or folder)
        """
        super().__init__(**kwargs)
        self.export_path = Path(export_path).expanduser()

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over Gmail messages."""
        import mailbox

        mbox_files = []
        if self.export_path.is_file() and self.export_path.suffix == ".mbox":
            mbox_files = [self.export_path]
        elif self.export_path.is_dir():
            mbox_files = list(self.export_path.glob("**/*.mbox"))

        for mbox_path in mbox_files:
            try:
                mbox = mailbox.mbox(str(mbox_path))

                for key, message in mbox.items():
                    # Extract email content
                    subject = message.get("subject", "")
                    sender = message.get("from", "")
                    to = message.get("to", "")
                    date_str = message.get("date", "")

                    # Parse date
                    timestamp = None
                    if date_str:
                        try:
                            from email.utils import parsedate_to_datetime
                            timestamp = parsedate_to_datetime(date_str)
                        except Exception:
                            pass

                    # Get body
                    body = ""
                    if message.is_multipart():
                        for part in message.walk():
                            if part.get_content_type() == "text/plain":
                                payload = part.get_payload(decode=True)
                                if payload:
                                    body = payload.decode("utf-8", errors="ignore")
                                    break
                    else:
                        payload = message.get_payload(decode=True)
                        if payload:
                            body = payload.decode("utf-8", errors="ignore")

                    if not body and not subject:
                        continue

                    # Combine subject and body
                    content = f"Subject: {subject}\n\n{body}" if subject else body

                    # Truncate very long emails
                    if len(content) > 10000:
                        content = content[:10000] + "..."

                    yield DataPoint(
                        content=content,
                        category=self.category,
                        source_type=self.source_type,
                        source_identifier=f"gmail_{key}",
                        timestamp=timestamp,
                        metadata={
                            "subject": subject,
                            "from": sender,
                            "to": to,
                            "labels": mbox_path.stem,
                        },
                        tags=["email", "gmail"],
                    )

            except Exception:
                # Skip problematic mbox files
                continue


class GmailAPISource(DataSource):
    """
    Adapter for Gmail via Google API (OAuth).

    Requires OAuth credentials from Google Cloud Console.
    First run will open browser for authentication.
    """

    source_type = "gmail_api"

    def __init__(
        self,
        credentials_path: str = "~/memory/data/persistent/gmail_credentials.json",
        token_path: str = "~/memory/data/persistent/gmail_token.json",
        max_emails: int = 10000,
        **kwargs,
    ):
        """
        Initialize Gmail API source.

        Args:
            credentials_path: Path to OAuth credentials JSON from Google Cloud
            token_path: Path to store auth token (created after first auth)
            max_emails: Maximum emails to fetch
        """
        super().__init__(**kwargs)
        self.credentials_path = Path(credentials_path).expanduser()
        self.token_path = Path(token_path).expanduser()
        self.max_emails = max_emails
        self._service = None

    @property
    def name(self) -> str:
        return "Gmail API"

    @property
    def category(self) -> DataCategory:
        return DataCategory.EMAIL

    async def is_available(self) -> bool:
        """Check if credentials exist."""
        return self.credentials_path.exists() or self.token_path.exists()

    async def estimate_count(self) -> int:
        """Estimate email count."""
        return self.max_emails  # Could query profile for actual count

    def _get_gmail_service(self):
        """Get authenticated Gmail service."""
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        creds = None

        # Load existing token
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.credentials_path.exists():
                    raise FileNotFoundError(
                        f"Gmail credentials not found at {self.credentials_path}\n"
                        "Please download OAuth credentials from Google Cloud Console:\n"
                        "1. Go to console.cloud.google.com\n"
                        "2. Create a project and enable Gmail API\n"
                        "3. Create OAuth 2.0 credentials (Desktop app)\n"
                        "4. Download and save as gmail_credentials.json"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save token for future use
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())

        return build('gmail', 'v1', credentials=creds)

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over Gmail messages."""
        try:
            service = self._get_gmail_service()
        except Exception as e:
            print(f"Gmail auth error: {e}")
            return

        # Get list of messages
        results = service.users().messages().list(
            userId='me',
            maxResults=min(500, self.max_emails)
        ).execute()

        messages = results.get('messages', [])
        next_page_token = results.get('nextPageToken')

        # Paginate through all messages
        while next_page_token and len(messages) < self.max_emails:
            results = service.users().messages().list(
                userId='me',
                pageToken=next_page_token,
                maxResults=min(500, self.max_emails - len(messages))
            ).execute()
            messages.extend(results.get('messages', []))
            next_page_token = results.get('nextPageToken')

        for msg_info in messages[:self.max_emails]:
            try:
                msg = service.users().messages().get(
                    userId='me',
                    id=msg_info['id'],
                    format='full'
                ).execute()

                # Extract headers
                headers = {h['name'].lower(): h['value'] for h in msg['payload'].get('headers', [])}
                subject = headers.get('subject', '')
                sender = headers.get('from', '')
                to = headers.get('to', '')
                date_str = headers.get('date', '')

                # Parse date
                timestamp = None
                if date_str:
                    try:
                        from email.utils import parsedate_to_datetime
                        timestamp = parsedate_to_datetime(date_str)
                    except Exception:
                        pass

                # Get body
                body = self._get_message_body(msg['payload'])
                if not body and not subject:
                    continue

                content = f"Subject: {subject}\n\n{body}" if subject else body
                if len(content) > 10000:
                    content = content[:10000] + "..."

                # Get labels
                labels = msg.get('labelIds', [])

                yield DataPoint(
                    content=content,
                    category=self.category,
                    source_type=self.source_type,
                    source_id=msg_info['id'],
                    original_date=timestamp,
                    raw_data={
                        "subject": subject,
                        "from": sender,
                        "to": to,
                        "labels": labels,
                        "thread_id": msg.get('threadId'),
                    },
                    topics=["email", "gmail"] + [l.lower() for l in labels if not l.startswith('CATEGORY_')],
                )

            except Exception:
                # Skip problematic messages
                continue

    def _get_message_body(self, payload) -> str:
        """Extract text body from message payload."""
        body = ""

        if 'body' in payload and payload['body'].get('data'):
            import base64
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')

        elif 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain':
                    if 'body' in part and part['body'].get('data'):
                        import base64
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                        break
                elif 'parts' in part:
                    # Nested multipart
                    body = self._get_message_body(part)
                    if body:
                        break

        return body


class AppleMailSource(DataSource):
    """Adapter for Apple Mail database."""

    source_type = "apple_mail"
    category = DataCategory.COMMUNICATION

    def __init__(
        self,
        mail_path: str = "~/Library/Mail",
        **kwargs,
    ):
        """
        Initialize Apple Mail source.

        Args:
            mail_path: Path to Mail library
        """
        super().__init__(**kwargs)
        self.mail_path = Path(mail_path).expanduser()

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over Apple Mail messages."""
        # Apple Mail stores messages in .emlx format
        for emlx_file in self.mail_path.glob("**/*.emlx"):
            try:
                with open(emlx_file, "rb") as f:
                    # .emlx files start with byte count on first line
                    first_line = f.readline()
                    try:
                        byte_count = int(first_line.strip())
                    except ValueError:
                        continue

                    # Read the email content
                    email_data = f.read(byte_count)

                import email
                from email.policy import default

                msg = email.message_from_bytes(email_data, policy=default)

                subject = msg.get("subject", "")
                sender = msg.get("from", "")
                to = msg.get("to", "")
                date_str = msg.get("date", "")

                # Parse date
                timestamp = None
                if date_str:
                    try:
                        from email.utils import parsedate_to_datetime
                        timestamp = parsedate_to_datetime(date_str)
                    except Exception:
                        pass

                # Get body
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_content()
                            break
                else:
                    body = msg.get_content()

                if not body and not subject:
                    continue

                content = f"Subject: {subject}\n\n{body}" if subject else body

                if len(content) > 10000:
                    content = content[:10000] + "..."

                yield DataPoint(
                    content=content,
                    category=self.category,
                    source_type=self.source_type,
                    source_identifier=str(emlx_file),
                    timestamp=timestamp,
                    metadata={
                        "subject": subject,
                        "from": sender,
                        "to": to,
                    },
                    tags=["email", "apple_mail"],
                )

            except Exception:
                continue
