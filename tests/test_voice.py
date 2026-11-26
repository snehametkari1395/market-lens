import pytest

from finance_research import voice
from finance_research.voice import VoiceError, spoken_summary


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code, self._payload, self.content = status, payload or {}, content
        self.ok = 200 <= status < 300
        self.text = ""

    def json(self):
        return self._payload


def test_transcribe_sends_ink_request_and_returns_text(monkeypatch) -> None:
    seen = {}

    def fake_post(url, headers, files, data, timeout):
        seen.update(url=url, headers=headers, files=files, data=data)
        return FakeResponse(payload={"type": "transcript", "text": " How did Apple do? "})

    monkeypatch.setattr(voice.requests, "post", fake_post)
    assert voice.transcribe(b"RIFF...", "sk_car_test", "de") == "How did Apple do?"
    assert seen["url"] == "https://api.cartesia.ai/stt"
    assert seen["headers"] == {"Authorization": "Bearer sk_car_test", "Cartesia-Version": "2026-08-14"}
    assert seen["data"] == {"model": "ink-whisper", "language": "de"}


def test_synthesize_requests_sonic_wav(monkeypatch) -> None:
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, body=json)
        return FakeResponse(content=b"WAVDATA")

    monkeypatch.setattr(voice.requests, "post", fake_post)
    assert voice.synthesize("Hello", "sk_car_test", "xx") == b"WAVDATA"
    assert seen["url"] == "https://api.cartesia.ai/tts/bytes"
    assert seen["body"]["model_id"] == "sonic-3.6"
    assert seen["body"]["output_format"]["container"] == "wav"
    assert seen["body"]["language"] == "en"  # unsupported codes fall back to English


def test_errors_are_friendly(monkeypatch) -> None:
    with pytest.raises(VoiceError, match="Cartesia API key"):
        voice.transcribe(b"x", "  ")
    monkeypatch.setattr(voice.requests, "post", lambda *a, **k: FakeResponse(status=401, payload={"message": "bad key"}))
    with pytest.raises(VoiceError, match="rejected the API key"):
        voice.synthesize("Hi", "sk_car_bad")
    monkeypatch.setattr(voice.requests, "post", lambda *a, **k: FakeResponse(payload={"text": ""}))
    with pytest.raises(VoiceError, match="No speech"):
        voice.transcribe(b"x", "sk_car_test")


def test_spoken_summary_drops_markdown_tables_and_citations() -> None:
    answer = (
        "## Apple revenue\n\nApple's revenue rose from $365.8B to $383.3B [1]. "
        "Growth slowed in the last year [1, 2]. See [the filing](https://sec.gov/x). Extra sentence here.\n\n"
        "| Year | Revenue |\n| --- | --- |\n| 2023 | 383.3 |\n"
    )
    summary = spoken_summary(answer)
    assert summary == "Apple's revenue rose from $365.8B to $383.3B. Growth slowed in the last year. See the filing."
    assert "[" not in summary and "|" not in summary and "http" not in summary
    assert len(spoken_summary("word " * 500, max_chars=100)) <= 101
