"""Authentication and token helpers."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import UTC, datetime, timedelta

PBKDF2_ROUNDS = 100_000


def _now_utc() -> datetime:
    return datetime.now(UTC)


def hash_password(password: str) -> str:
    """Return stored representation: base64(salt)$base64(hash)."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, digest_hex = stored_hash.split("$", maxsplit=1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except ValueError:
        return False

    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return hmac.compare_digest(candidate, expected)


def generate_token() -> str:
    return secrets.token_urlsafe(32)


def expiry_iso(hours: int = 24) -> str:
    return (_now_utc() + timedelta(hours=hours)).isoformat()


def is_expired(expires_at: str) -> bool:
    return datetime.fromisoformat(expires_at) <= _now_utc()
