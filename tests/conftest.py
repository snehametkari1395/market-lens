import pytest


@pytest.fixture(autouse=True)
def _isolated_rate_limit_file(tmp_path, monkeypatch):
    """Keep the app's saved spending (data/ratelimit_state.json) out of tests, and tests out of it."""
    monkeypatch.setenv("MARKET_LENS_RATELIMIT_FILE", str(tmp_path / "ratelimit_state.json"))
