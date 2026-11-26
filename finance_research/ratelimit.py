"""Rate limiting for model calls, and API key checks."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import requests


@dataclass
class RateLimits:
    per_minute: int = 5
    per_hour: int = 40
    daily_budget_usd: float = 1.00


class RateLimitExceeded(RuntimeError):
    """Raised when a request would exceed a limit. `retry_after` is in seconds."""

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class _Bucket:
    requests: deque = field(default_factory=deque)  # timestamps
    spend: deque = field(default_factory=deque)  # (timestamp, usd)


class RateLimiter:
    """Sliding-window limiter keyed by user. Thread-safe so it can be shared across sessions."""

    def __init__(self, limits: RateLimits | None = None, clock=time.time) -> None:
        self.limits = limits or RateLimits()
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def _bucket(self, user: str) -> _Bucket:
        bucket = self._buckets.setdefault(user, _Bucket())
        now = self._clock()
        while bucket.requests and now - bucket.requests[0] >= 3600:
            bucket.requests.popleft()
        while bucket.spend and now - bucket.spend[0][0] >= 86400:
            bucket.spend.popleft()
        return bucket

    def check(self, user: str, limits: RateLimits | None = None) -> None:
        """Record a request, or raise RateLimitExceeded without recording it."""
        limits = limits or self.limits
        with self._lock:
            bucket = self._bucket(user)
            now = self._clock()
            last_minute = [stamp for stamp in bucket.requests if now - stamp < 60]
            if len(last_minute) >= limits.per_minute:
                wait = 60 - (now - last_minute[0])
                raise RateLimitExceeded(
                    f"Rate limit: {limits.per_minute} questions per minute. Try again in {wait:.0f} s.", wait
                )
            if len(bucket.requests) >= limits.per_hour:
                wait = 3600 - (now - bucket.requests[0])
                raise RateLimitExceeded(
                    f"Rate limit: {limits.per_hour} questions per hour. Try again in {wait / 60:.0f} min.", wait
                )
            spent = sum(amount for _, amount in bucket.spend)
            if spent >= limits.daily_budget_usd:
                wait = 86400 - (now - bucket.spend[0][0]) if bucket.spend else 86400
                raise RateLimitExceeded(
                    f"Daily spending limit of ${limits.daily_budget_usd:.2f} reached (${spent:.4f} used). "
                    "Raise it under API keys & limits, or try again later.",
                    wait,
                )
            bucket.requests.append(now)

    def record_cost(self, user: str, usd: float | None) -> None:
        if usd:
            with self._lock:
                self._bucket(user).spend.append((self._clock(), float(usd)))

    def status(self, user: str, limits: RateLimits | None = None) -> dict[str, float]:
        limits = limits or self.limits
        with self._lock:
            bucket = self._bucket(user)
            now = self._clock()
            return {
                "last_minute": sum(1 for stamp in bucket.requests if now - stamp < 60),
                "last_hour": len(bucket.requests),
                "spent_today": sum(amount for _, amount in bucket.spend),
                "per_minute": limits.per_minute,
                "per_hour": limits.per_hour,
                "daily_budget_usd": limits.daily_budget_usd,
            }


# --- API key management -----------------------------------------------------------


def mask_key(key: str) -> str:
    key = key.strip()
    if not key:
        return "not set"
    if len(key) <= 10:
        return "•" * len(key)
    return f"{key[:5]}…{key[-4:]}"


def openrouter_balance(key: str, timeout: int = 10) -> dict[str, float | None] | None:
    """Spend and limits for an OpenRouter key, or None if it can't be read.

    `key_*` come from /api/v1/key (works with any key). `account_*` come from
    /api/v1/credits, which OpenRouter only answers for some keys; they are None otherwise.
    """
    key = key.strip()
    if not key:
        return None
    headers = {"Authorization": f"Bearer {key}"}
    try:
        response = requests.get("https://openrouter.ai/api/v1/key", headers=headers, timeout=timeout)
        if not response.ok:
            return None
        data = response.json().get("data", {})
    except (requests.RequestException, ValueError):
        return None

    def number(value: object) -> float | None:
        return float(value) if isinstance(value, (int, float)) else None

    result: dict[str, float | None] = {
        "key_usage": number(data.get("usage")),
        "key_limit": number(data.get("limit")),
        "key_remaining": number(data.get("limit_remaining")),
        "account_credits": None,
        "account_remaining": None,
    }
    try:
        credits = requests.get("https://openrouter.ai/api/v1/credits", headers=headers, timeout=timeout)
        if credits.ok:
            totals = credits.json().get("data", {})
            bought, used = number(totals.get("total_credits")), number(totals.get("total_usage"))
            if bought is not None and used is not None:
                result["account_credits"] = bought
                result["account_remaining"] = bought - used
    except (requests.RequestException, ValueError):
        pass
    return result


def verify_api_key(provider: str, key: str, timeout: int = 15) -> tuple[bool, str]:
    """Check a key with a free, read-only request. Returns (ok, message)."""
    key = key.strip()
    if not key:
        return False, "No key entered."
    try:
        if provider == "openrouter":
            response = requests.get(
                "https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}, timeout=timeout
            )
            if response.ok:
                data = response.json().get("data", {})
                limit = data.get("limit")
                usage = data.get("usage")
                detail = f"used ${usage:.2f}" if isinstance(usage, (int, float)) else "valid"
                if isinstance(limit, (int, float)):
                    detail += f" of ${limit:.2f} limit"
                return True, f"Key works ({detail})."
        elif provider == "openai":
            response = requests.get(
                "https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=timeout
            )
            if response.ok:
                return True, "Key works."
        elif provider == "anthropic":
            import anthropic

            try:
                anthropic.Anthropic(api_key=key, max_retries=0, timeout=timeout).models.list(limit=1)
                return True, "Key works."
            except anthropic.AuthenticationError:
                return False, "Anthropic rejected this key (401)."
            except anthropic.PermissionDeniedError:
                return False, "This key doesn't have permission to use the API (403)."
            except anthropic.APIConnectionError:
                return False, "Couldn't reach Anthropic. Check your internet connection."
            except anthropic.APIStatusError as error:
                return False, f"Anthropic returned an error ({error.status_code})."
        elif provider == "alphavantage":
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={"function": "GLOBAL_QUOTE", "symbol": "IBM", "apikey": key},
                timeout=timeout,
            )
            payload = response.json() if response.ok else {}
            if payload.get("Global Quote"):
                return True, "Key works."
            message = payload.get("Note") or payload.get("Information") or payload.get("Error Message")
            return False, (message or "Alpha Vantage didn't accept this key.")[:200]
        else:
            return False, f"Unknown provider: {provider}."
        return False, f"Rejected ({response.status_code}). Check the key."
    except requests.RequestException as error:
        return False, f"Couldn't check the key: {type(error).__name__}."
