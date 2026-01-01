"""MFA (TOTP) management using pyotp."""

import base64
import io
import secrets
import string

import pyotp
import qrcode
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class MFAManager:
    """Manages TOTP-based multi-factor authentication."""

    def __init__(self, encryption_key: bytes):
        """Initialize with an encryption key for storing secrets."""
        self.fernet = Fernet(encryption_key)
        self.issuer = "PersonalMemorySystem"

    @staticmethod
    def generate_encryption_key(master_password: str, salt: bytes) -> bytes:
        """Derive an encryption key from a master password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        return base64.urlsafe_b64encode(kdf.derive(master_password.encode()))

    @staticmethod
    def generate_salt() -> bytes:
        """Generate a random salt for key derivation."""
        return secrets.token_bytes(16)

    def generate_secret(self) -> str:
        """Generate a new TOTP secret."""
        return pyotp.random_base32()

    def encrypt_secret(self, secret: str) -> bytes:
        """Encrypt a TOTP secret for storage."""
        return self.fernet.encrypt(secret.encode())

    def decrypt_secret(self, encrypted: bytes) -> str:
        """Decrypt a stored TOTP secret."""
        return self.fernet.decrypt(encrypted).decode()

    def get_provisioning_uri(self, secret: str, username: str) -> str:
        """Get the otpauth:// URI for QR code generation."""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=username, issuer_name=self.issuer)

    def generate_qr_code_base64(self, provisioning_uri: str) -> str:
        """Generate a QR code as base64-encoded PNG."""
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(provisioning_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode()

    def verify_totp(self, secret: str, code: str) -> bool:
        """Verify a TOTP code against a secret."""
        totp = pyotp.TOTP(secret)
        # Allow 1 window before and after for clock drift
        return totp.verify(code, valid_window=1)

    def generate_backup_codes(self, count: int = 10) -> list[str]:
        """Generate one-time backup codes."""
        codes = []
        # Generate codes in format: XXXX-XXXX
        alphabet = string.ascii_uppercase + string.digits
        for _ in range(count):
            part1 = "".join(secrets.choice(alphabet) for _ in range(4))
            part2 = "".join(secrets.choice(alphabet) for _ in range(4))
            codes.append(f"{part1}-{part2}")
        return codes

    def hash_backup_code(self, code: str) -> str:
        """Hash a backup code for storage."""
        # Normalize code (remove dashes, uppercase)
        normalized = code.replace("-", "").upper()
        # Use the password manager for hashing
        from .password import get_password_manager

        return get_password_manager().hash(normalized)

    def verify_backup_code(self, code: str, hashed: str) -> bool:
        """Verify a backup code against its hash."""
        normalized = code.replace("-", "").upper()
        from .password import get_password_manager

        return get_password_manager().verify(normalized, hashed)


# Module-level MFA manager (initialized lazily)
_mfa_manager: MFAManager | None = None


def get_mfa_manager() -> MFAManager:
    """Get the MFA manager instance."""
    global _mfa_manager
    if _mfa_manager is None:
        raise RuntimeError("MFA manager not initialized. Call init_mfa_manager first.")
    return _mfa_manager


def init_mfa_manager(encryption_key: bytes) -> MFAManager:
    """Initialize the MFA manager with an encryption key."""
    global _mfa_manager
    _mfa_manager = MFAManager(encryption_key)
    return _mfa_manager
