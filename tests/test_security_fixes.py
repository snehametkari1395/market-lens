import socket

import pytest

from finance_research import mcp_tools
from finance_research.ratelimit import RateLimiter, RateLimitExceeded, RateLimits


def fake_dns(mapping):
    def getaddrinfo(host, port, proto=0):
        if host not in mapping:
            raise socket.gaierror("not found")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port)) for address in mapping[host]]
    return getaddrinfo


@pytest.mark.parametrize(
    "url",
    [
        "https://10.0.0.5/mcp",
        "https://192.168.1.20/mcp",
        "https://169.254.169.254/latest/meta-data",  # cloud metadata endpoint
        "https://[::1]/mcp",
        "https://[::ffff:127.0.0.1]/mcp",
        "https://intranet.example/mcp",  # resolves to a private address
        "https://mixed.example/mcp",  # one public and one private address
        "https://missing.example/mcp",
    ],
)
def test_private_or_unknown_mcp_hosts_are_rejected(monkeypatch, url) -> None:
    monkeypatch.setattr(mcp_tools.socket, "getaddrinfo", fake_dns({
        "intranet.example": ["10.1.2.3"], "mixed.example": ["93.184.216.34", "172.16.0.1"],
    }))
    with pytest.raises(ValueError, match="public"):
        mcp_tools.validate_mcp_url(url)


def test_public_mcp_host_and_trusted_localhost_are_allowed(monkeypatch) -> None:
    monkeypatch.setattr(mcp_tools.socket, "getaddrinfo", fake_dns({"mcp.deepwiki.com": ["104.21.10.10"]}))
    assert mcp_tools.validate_mcp_url("https://mcp.deepwiki.com/mcp") == "https://mcp.deepwiki.com/mcp"
    assert mcp_tools.validate_mcp_url("http://127.0.0.1:8765/mcp", allow_local=True) == "http://127.0.0.1:8765/mcp"


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def test_daily_spend_survives_a_restart_and_expires(tmp_path) -> None:
    path, clock = tmp_path / "limits.json", Clock()
    limits = RateLimits(per_minute=100, per_hour=100, daily_budget_usd=1.0)
    first = RateLimiter(clock=clock, state_path=path)
    first.record_cost("alice", 0.6)
    first.record_cost("alice", 0.5)

    restarted = RateLimiter(clock=clock, state_path=path)  # e.g. the app was rebooted
    with pytest.raises(RateLimitExceeded, match="Daily spending limit"):
        restarted.check("alice", limits)
    restarted.check("bob", limits)  # other users unaffected

    clock.now += 86401  # a day later the old spend no longer counts
    RateLimiter(clock=clock, state_path=path).check("alice", limits)


def test_damaged_state_file_is_ignored_and_idle_users_are_pruned(tmp_path) -> None:
    path = tmp_path / "limits.json"
    path.write_text("{not json", encoding="utf-8")
    clock = Clock()
    limiter = RateLimiter(clock=clock, state_path=path)
    for index in range(1005):
        limiter.check(f"user{index}", RateLimits(per_minute=5, per_hour=5, daily_budget_usd=1))
    clock.now += 3601
    limiter.check("late", RateLimits())
    assert len(limiter._buckets) < 10
