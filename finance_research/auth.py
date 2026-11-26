"""Local user accounts and saved preferences.

Passwords are stored only as salted PBKDF2-SHA256 hashes. API keys are never saved.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PBKDF2_ITERATIONS = 310_000
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")
DEFAULT_PREFERENCES: dict[str, Any] = {
    "display_name": "",
    "watchlist": ["AAPL", "MSFT"],
    "default_metric": "revenue",
    "years": 5,
    "provider": "openrouter",
    "model": "",
    "mcp_enabled": False,
    "language": "en",
    "search_strategy": "hybrid",
    "tool_source": "builtin",
}
_lock = threading.Lock()


class AuthError(ValueError):
    pass


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations))
    return hmac.compare_digest(digest.hex(), digest_hex)


class UserStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"users": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def create_user(self, username: str, password: str, display_name: str = "") -> dict[str, Any]:
        username = username.strip()
        if not USERNAME_PATTERN.match(username):
            raise AuthError("Usernames are 3–32 characters: letters, numbers, dot, dash or underscore.")
        if len(password) < 8:
            raise AuthError("Use a password of at least 8 characters.")
        with _lock:
            data = self._load()
            if username.lower() in data["users"]:
                raise AuthError("That username is taken.")
            data["users"][username.lower()] = {
                "username": username,
                "password_hash": hash_password(password),
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "preferences": {**DEFAULT_PREFERENCES, "display_name": display_name.strip() or username},
            }
            self._save(data)
        return self.public_profile(username)

    def authenticate(self, username: str, password: str) -> dict[str, Any]:
        user = self._load()["users"].get(username.strip().lower())
        # Hash even for unknown users so response time doesn't reveal which usernames exist.
        stored = user["password_hash"] if user else hash_password("timing-equaliser")
        if not verify_password(password, stored) or not user:
            raise AuthError("Wrong username or password.")
        return self.public_profile(username)

    def public_profile(self, username: str) -> dict[str, Any]:
        user = self._load()["users"][username.strip().lower()]
        return {
            "username": user["username"],
            "created": user["created"],
            "preferences": {**DEFAULT_PREFERENCES, **user.get("preferences", {})},
        }

    def admin_usernames(self) -> set[str]:
        """Lower-case usernames with admin rights.

        MARKET_LENS_ADMINS (comma-separated) wins when set; otherwise the first account
        ever created on this machine is the admin.
        """
        configured = {name.strip().lower() for name in os.getenv("MARKET_LENS_ADMINS", "").split(",") if name.strip()}
        if configured:
            return configured
        users = self._load()["users"]
        if not users:
            return set()
        first = min(users.items(), key=lambda item: item[1].get("created", ""))
        return {first[0]}

    def is_admin(self, username: str) -> bool:
        return bool(username) and username.strip().lower() in self.admin_usernames()

    def exists(self, username: str) -> bool:
        return username.strip().lower() in self._load()["users"]

    def update_preferences(self, username: str, preferences: dict[str, Any]) -> dict[str, Any]:
        allowed = {key: value for key, value in preferences.items() if key in DEFAULT_PREFERENCES}
        with _lock:
            data = self._load()
            user = data["users"][username.strip().lower()]
            user["preferences"] = {**DEFAULT_PREFERENCES, **user.get("preferences", {}), **allowed}
            self._save(data)
        return self.public_profile(username)
