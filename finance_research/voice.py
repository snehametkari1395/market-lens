"""Voice input and output through Cartesia: Ink (speech-to-text) and Sonic (text-to-speech).

The research assistant stays in charge of answers. This module only turns speech into a
question and a short, citation-free summary of the answer into speech.
API reference: https://docs.cartesia.ai/api-reference (version 2026-08-14).
"""

from __future__ import annotations

import re

import requests

CARTESIA_BASE_URL = "https://api.cartesia.ai"
CARTESIA_VERSION = "2026-08-14"
TTS_MODEL = "sonic-3.6"
STT_MODEL = "ink-whisper"  # the only model the batch /stt endpoint accepts
DEFAULT_VOICE_ID = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"  # "Skylar", from Cartesia's Sonic 3.6 docs
VOICE_LANGUAGES = ("en", "es", "fr", "de", "pt", "hi")  # the app's interface languages; all supported by Sonic 3.6 and ink-whisper


class VoiceError(RuntimeError):
    """Raised when Cartesia can't transcribe or speak; the message is safe to show to users."""


def _headers(api_key: str) -> dict[str, str]:
    key = api_key.strip()
    if not key:
        raise VoiceError("Add a Cartesia API key to use voice.")
    return {"Authorization": f"Bearer {key}", "Cartesia-Version": CARTESIA_VERSION}


def _raise_for_status(response: requests.Response, action: str) -> None:
    if response.ok:
        return
    try:
        detail = response.json().get("message") or response.json().get("title") or ""
    except ValueError:
        detail = response.text[:200]
    if response.status_code in (401, 403):
        raise VoiceError(f"Cartesia rejected the API key while trying to {action} ({response.status_code}).")
    if response.status_code == 402:
        raise VoiceError("Your Cartesia credits are used up.")
    raise VoiceError(f"Cartesia couldn't {action} ({response.status_code}). {detail}".strip())


def _language(language: str) -> str:
    return language if language in VOICE_LANGUAGES else "en"


def transcribe(audio: bytes, api_key: str, language: str = "en", filename: str = "question.wav", timeout: int = 120) -> str:
    """Speech to text with Ink. `audio` is a recording in wav, webm, mp3, ogg, m4a or flac."""
    if not audio:
        raise VoiceError("The recording was empty. Try again.")
    try:
        response = requests.post(
            f"{CARTESIA_BASE_URL}/stt",
            headers=_headers(api_key),
            files={"file": (filename, audio)},
            data={"model": STT_MODEL, "language": _language(language)},
            timeout=timeout,
        )
    except requests.RequestException as error:
        raise VoiceError(f"Couldn't reach Cartesia: {type(error).__name__}.") from None
    _raise_for_status(response, "transcribe the recording")
    text = str(response.json().get("text") or "").strip()
    if not text:
        raise VoiceError("No speech was recognised. Speak a little closer to the microphone and try again.")
    return text


def synthesize(text: str, api_key: str, language: str = "en", voice_id: str = DEFAULT_VOICE_ID, timeout: int = 60) -> bytes:
    """Text to speech with Sonic. Returns WAV audio."""
    if not text.strip():
        raise VoiceError("Nothing to say.")
    body = {
        "model_id": TTS_MODEL,
        "transcript": text,
        "voice": {"id": voice_id},
        "output_format": {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 44100},
        "language": _language(language),
    }
    try:
        response = requests.post(f"{CARTESIA_BASE_URL}/tts/bytes", headers=_headers(api_key), json=body, timeout=timeout)
    except requests.RequestException as error:
        raise VoiceError(f"Couldn't reach Cartesia: {type(error).__name__}.") from None
    _raise_for_status(response, "generate speech")
    return response.content


_MARKDOWN_NOISE = [
    (re.compile(r"```.*?```", re.S), " "),  # code blocks
    (re.compile(r"^\s*\|.*\|\s*$", re.M), " "),  # table rows
    (re.compile(r"^\s{0,3}#{1,6}\s.*$", re.M), " "),  # headings: dropped, they don't read as sentences
    (re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]"), ""),  # citation markers like [1] or [1, 2]
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),  # links -> their text
    (re.compile(r"https?://\S+"), ""),
    (re.compile(r"^\s*[-*+]\s+", re.M), ""),  # bullets
    (re.compile(r"[*_`>]+"), ""),
]


def spoken_summary(answer: str, max_sentences: int = 3, max_chars: int = 450) -> str:
    """The first few sentences of an answer, without markdown, tables or citation markers.

    Tables and footnotes don't work read aloud; the full cited answer stays on screen.
    """
    text = answer or ""
    for pattern, replacement in _MARKDOWN_NOISE:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = ""
    for sentence in sentences[:max_sentences]:
        if summary and len(summary) + len(sentence) + 1 > max_chars:
            break
        summary = f"{summary} {sentence}".strip()
    if len(summary) > max_chars:
        summary = summary[: max_chars].rsplit(" ", 1)[0] + "…"
    return summary
