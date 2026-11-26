"""Automated knowledge-base updates.

Two ways to run them without anyone clicking a button:

* In the app: a background thread in the Streamlit server checks SEC EDGAR on a
  fixed interval, whether or not a browser is open.
* From the operating system: `python -m finance_research.scheduler` runs one
  update and exits, for Windows Task Scheduler or cron. It reads credentials
  from environment variables.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


@dataclass
class SchedulerConfig:
    interval_hours: float
    sec_email: str
    openrouter_key: str
    tickers: list[str] | None = None  # None means every company already indexed


def run_update(config: SchedulerConfig, data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    """One update pass over the knowledge base. Returns the per-company report."""
    from finance_research.rag import create_vector_store, index_summary, refresh_knowledge_base
    from finance_research.sec_client import validate_sec_user_agent

    vector_dir = data_dir / "chroma"
    tickers = config.tickers or sorted({row["ticker"] for row in index_summary(vector_dir)})
    if not tickers:
        return []
    return refresh_knowledge_base(
        tickers,
        validate_sec_user_agent(config.sec_email),
        create_vector_store(config.openrouter_key, vector_dir),
        data_dir / "kb_refresh_status.json",
    )


class KnowledgeBaseScheduler:
    """Background thread that runs `run_update` every `interval_hours`.

    Credentials stay in memory only. The scheduler stops when the server stops.
    """

    def __init__(
        self,
        on_event: Callable[..., Any] | None = None,
        update: Callable[[SchedulerConfig], list[dict[str, Any]]] = run_update,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._config: SchedulerConfig | None = None
        self._update = update
        self._clock = clock
        self._on_event = on_event or (lambda *args, **kwargs: None)
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.next_run: float | None = None
        self.last_run: dict[str, Any] | None = None
        self.running = False

    @property
    def enabled(self) -> bool:
        return self._config is not None

    @property
    def interval_hours(self) -> float | None:
        return self._config.interval_hours if self._config else None

    def start(self, config: SchedulerConfig, run_now: bool = False) -> None:
        if config.interval_hours <= 0:
            raise ValueError("The update interval must be positive.")
        if not config.sec_email.strip() or not config.openrouter_key.strip():
            raise ValueError("Automated updates need an SEC contact email and an OpenRouter key.")
        with self._lock:
            self._config = config
            self.next_run = self._clock() + (0 if run_now else config.interval_hours * 3600)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="kb-scheduler", daemon=True)
                self._thread.start()
        self._wake.set()
        self._on_event("kb_scheduler_started", interval_hours=config.interval_hours)

    def stop(self) -> None:
        with self._lock:
            self._config = None
            self.next_run = None
        self._wake.set()
        self._on_event("kb_scheduler_stopped")

    def run_once(self) -> list[dict[str, Any]]:
        config = self._config
        if config is None:
            return []
        self.running = True
        started = self._clock()
        try:
            report = self._update(config)
            self.last_run = {
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "companies": len(report),
                "new_filings": sum(len(item.get("new_filings", [])) for item in report),
                "chunks_added": sum(item.get("chunks_added", 0) for item in report),
                "errors": sum(1 for item in report if item.get("error")),
                "duration_s": round(self._clock() - started, 1),
                "ok": True,
            }
            self._on_event("kb_refresh", source="scheduler", **{k: v for k, v in self.last_run.items() if k != "ok"})
            return report
        except Exception as error:  # keep the thread alive; report the failure
            self.last_run = {
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "ok": False,
                "error": f"{type(error).__name__}: {error}"[:300],
            }
            self._on_event("kb_refresh_error", level="error", source="scheduler", error=self.last_run["error"])
            return []
        finally:
            self.running = False

    def _loop(self) -> None:
        while True:
            with self._lock:
                config, due = self._config, self.next_run
            if config is None or due is None:
                self._wake.wait(timeout=60)
                self._wake.clear()
                continue
            wait = due - self._clock()
            if wait > 0:
                self._wake.wait(timeout=min(wait, 60))
                self._wake.clear()
                continue
            self.run_once()
            with self._lock:
                if self._config is not None:
                    self.next_run = self._clock() + self._config.interval_hours * 3600

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "interval_hours": self.interval_hours,
            "next_run": datetime.fromtimestamp(self.next_run, timezone.utc).isoformat(timespec="seconds")
            if self.next_run
            else None,
            "running": self.running,
            "last_run": self.last_run,
        }


def task_scheduler_command(python_exe: str, project_dir: str, hour: int = 7) -> str:
    """A Windows `schtasks` command that runs the daily update at the given hour."""
    action = f'cmd /c cd /d "{project_dir}" && "{python_exe}" -m finance_research.scheduler'
    return f'schtasks /Create /SC DAILY /ST {hour:02d}:00 /TN "MarketLens KB update" /TR "{action}" /F'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check SEC EDGAR for new 10-K filings and index them.")
    parser.add_argument("--tickers", nargs="*", help="Companies to update (default: every indexed company).")
    args = parser.parse_args(argv)
    config = SchedulerConfig(
        interval_hours=24,
        sec_email=os.getenv("SEC_USER_AGENT_EMAIL", ""),
        openrouter_key=os.getenv("OPENROUTER_API_KEY", ""),
        tickers=[ticker.upper() for ticker in args.tickers] if args.tickers else None,
    )
    if not config.sec_email or not config.openrouter_key:
        print("Set SEC_USER_AGENT_EMAIL and OPENROUTER_API_KEY environment variables first.", file=sys.stderr)
        return 2

    from finance_research.monitoring import Monitor

    monitor = Monitor(DATA_DIR / "logs")
    try:
        report = run_update(config)
    except Exception as error:
        monitor.event("kb_refresh_error", level="error", source="task-scheduler", error=str(error))
        print(f"Update failed: {error}", file=sys.stderr)
        return 1
    monitor.event(
        "kb_refresh",
        source="task-scheduler",
        companies=len(report),
        new_filings=sum(len(item["new_filings"]) for item in report),
        chunks_added=sum(item["chunks_added"] for item in report),
        errors=sum(1 for item in report if item["error"]),
    )
    for item in report:
        added = ", ".join(item["new_filings"]) or "nothing new"
        print(f"{item['ticker']}: latest {item['latest_filing']} - {added}" + (f" (error: {item['error']})" if item["error"] else ""))
    if not report:
        print("No indexed companies yet. Index filings in the app first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
