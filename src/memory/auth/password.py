"""Password hashing and verification using bcrypt directly."""

import bcrypt


class PasswordManager:
    """Manages password hashing and verification."""

    def __init__(self, rounds: int = 12):
        self.rounds = rounds

    def hash(self, password: str) -> str:
        """Hash a password using bcrypt."""
        password_bytes = password.encode("utf-8")
        salt = bcrypt.gensalt(rounds=self.rounds)
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode("utf-8")

    def verify(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        try:
            password_bytes = password.encode("utf-8")
            hashed_bytes = hashed.encode("utf-8")
            return bcrypt.checkpw(password_bytes, hashed_bytes)
        except Exception:
            return False

    def needs_rehash(self, hashed: str) -> bool:
        """Check if a password hash needs to be upgraded (not implemented with bcrypt directly)."""
        return False


# Singleton instance
_password_manager: PasswordManager | None = None


def get_password_manager() -> PasswordManager:
    """Get the singleton password manager instance."""
    global _password_manager
    if _password_manager is None:
        _password_manager = PasswordManager()
    return _password_manager
