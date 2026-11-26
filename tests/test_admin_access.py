from finance_research.auth import UserStore


def test_first_account_is_admin_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MARKET_LENS_ADMINS", raising=False)
    store = UserStore(tmp_path / "users.json")
    assert store.admin_usernames() == set()
    store.create_user("Owner", "password123")
    store.create_user("visitor", "password123")
    assert store.is_admin("owner")
    assert store.is_admin("OWNER")
    assert not store.is_admin("visitor")
    assert not store.is_admin("")


def test_admin_env_variable_overrides_first_account(tmp_path, monkeypatch) -> None:
    store = UserStore(tmp_path / "users.json")
    store.create_user("owner", "password123")
    store.create_user("visitor", "password123")
    monkeypatch.setenv("MARKET_LENS_ADMINS", " Visitor , someone-else ")
    assert store.is_admin("visitor")
    assert not store.is_admin("owner")


def test_exists_is_case_insensitive(tmp_path) -> None:
    store = UserStore(tmp_path / "users.json")
    store.create_user("Owner", "password123")
    assert store.exists("owner")
    assert not store.exists("my-secret-password")


def test_openrouter_balance_reads_key_limits_and_skips_forbidden_credits(monkeypatch) -> None:
    from finance_research import ratelimit

    class Response:
        def __init__(self, ok, payload):
            self.ok, self._payload = ok, payload

        def json(self):
            return self._payload

    def fake_get(url, headers, timeout):
        if url.endswith("/key"):
            return Response(True, {"data": {"usage": 0.34, "limit": 10, "limit_remaining": 9.66}})
        return Response(False, {})  # /credits answers 403 for ordinary keys

    monkeypatch.setattr(ratelimit.requests, "get", fake_get)
    balance = ratelimit.openrouter_balance("sk-or-test")
    assert balance["key_remaining"] == 9.66 and balance["key_limit"] == 10.0
    assert balance["account_remaining"] is None
    assert ratelimit.openrouter_balance("  ") is None
