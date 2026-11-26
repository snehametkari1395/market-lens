"""Logging and monitoring: a rotating text log plus a JSON-lines event log for the dashboard."""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),  # OpenAI / OpenRouter / Anthropic style keys
    re.compile(r"(?i)(api[_-]?key|apikey|token|password)(\"?\s*[:=]\s*\"?)[^\s\"',&]+"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),  # email addresses
]
_lock = threading.Lock()


def redact(text: str) -> str:
    text = _SECRET_PATTERNS[0].sub("[redacted-key]", text)
    text = _SECRET_PATTERNS[1].sub(lambda m: f"{m.group(1)}{m.group(2)}[redacted]", text)
    return _SECRET_PATTERNS[2].sub("[redacted-email]", text)


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


class Monitor:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / "events.jsonl"
        self.logger = logging.getLogger("market_lens")
        if not any(isinstance(h, RotatingFileHandler) for h in self.logger.handlers):
            handler = RotatingFileHandler(self.log_dir / "market_lens.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def event(self, kind: str, level: str = "info", **fields: Any) -> dict[str, Any]:
        """Record one event in both logs. Secrets and emails are redacted."""
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind, "level": level}
        record.update(_redact_value(fields))
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        getattr(self.logger, level if level in {"info", "warning", "error"} else "info")(f"{kind} {line}")
        return record

    def read_events(self, since_hours: float | None = None, limit: int = 5000) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours) if since_hours else None
        events = []
        with self.events_path.open(encoding="utf-8") as handle:
            for line in handle.readlines()[-limit:]:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if cutoff and datetime.fromisoformat(record["ts"]) < cutoff:
                    continue
                events.append(record)
        return events


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline numbers for the monitoring dashboard."""
    questions = [e for e in events if e.get("kind") == "chat_answer"]
    failures = [e for e in events if e.get("kind") == "chat_error"]
    blocked = [e for e in events if e.get("kind") in {"injection_blocked", "rate_limited"}]
    latencies = sorted(e.get("latency_s", 0) for e in questions)
    attempts = len(questions) + len(failures)
    tool_counts: dict[str, int] = {}
    for event in questions:
        for name in event.get("tools", []) or []:
            tool_counts[name] = tool_counts.get(name, 0) + 1
    return {
        "answers": len(questions),
        "errors": len(failures),
        "blocked": len(blocked),
        "error_rate": (len(failures) / attempts) if attempts else 0.0,
        "median_latency_s": latencies[len(latencies) // 2] if latencies else 0.0,
        "p90_latency_s": latencies[min(len(latencies) - 1, int(len(latencies) * 0.9))] if latencies else 0.0,
        "tokens": sum(int(e.get("total_tokens") or 0) for e in questions),
        "cost_usd": sum(float(e.get("cost_usd") or 0) for e in questions),
        "tool_counts": tool_counts,
        "figures_checked": sum(int(e.get("figures_checked") or 0) for e in questions),
        "figures_unverified": sum(int(e.get("figures_unverified") or 0) for e in questions),
        "figure_check_rate": (
            1 - sum(int(e.get("figures_unverified") or 0) for e in questions) / checked
            if (checked := sum(int(e.get("figures_checked") or 0) for e in questions))
            else None
        ),
    }
