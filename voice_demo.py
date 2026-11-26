"""Market Lens voice demo: ask the research assistant out loud and hear a short answer.

A standalone page, separate from streamlit_app.py, so it can be tried and thrown away
without touching the main app. Run it with:

    .\\.venv\\Scripts\\python.exe -m streamlit run voice_demo.py --server.port 8502

Speech: Cartesia Ink (speech-to-text) and Sonic (text-to-speech).
Answers: the same research assistant, tools, safety checks and citations as the main app.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from finance_research.agent import run_research_agent
from finance_research.conversation import extract_citations, format_citations_markdown
from finance_research.providers import PROVIDERS
from finance_research.security import PromptInjectionError
from finance_research.sec_client import validate_sec_user_agent
from finance_research.tools import build_tools
from finance_research.voice import VOICE_LANGUAGES, VoiceError, spoken_summary, synthesize, transcribe

LANGUAGE_NAMES = {"en": "English", "es": "Español", "fr": "Français", "de": "Deutsch", "pt": "Português", "hi": "हिन्दी"}
EXAMPLES = [
    "How did Apple's revenue change over the last three years?",
    "Compare Microsoft and Alphabet net income for the last five years.",
    "What is Nvidia's net margin for its latest fiscal year?",
]

st.set_page_config(page_title="Market Lens · Voice demo", page_icon=":material/mic:", layout="centered")


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def _tool_results(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    args_by_id = {call["id"]: call.get("args", {}) for m in messages if isinstance(m, AIMessage) for call in m.tool_calls}
    results = []
    for message in messages:
        if isinstance(message, ToolMessage):
            try:
                content = json.loads(str(message.content))
            except json.JSONDecodeError:
                content = {"result": str(message.content)}
            results.append({"name": message.name, "args": args_by_id.get(message.tool_call_id, {}), "content": content})
    return results


with st.sidebar:
    st.title("Voice demo")
    st.caption("Keys stay in this browser session. Saved Windows settings are used when present.")
    cartesia_key = st.text_input("Cartesia API key", value=_env("CARTESIA_API_KEY"), type="password",
                                 help="From play.cartesia.ai → API keys. Starts with sk_car_.")
    openrouter_key = st.text_input("OpenRouter API key", value=_env("OPENROUTER_API_KEY"), type="password")
    sec_email = st.text_input("SEC contact email", value=_env("SEC_USER_AGENT_EMAIL"),
                              help="The SEC asks every app to identify itself with a contact email.")
    model = st.selectbox("Model", PROVIDERS["openrouter"].models, index=0)
    language = st.selectbox("Language", VOICE_LANGUAGES, format_func=LANGUAGE_NAMES.get)
    speak_answers = st.toggle("Speak the answer", value=True)

st.title(":material/mic: Ask Market Lens out loud")
st.write("Press the microphone, ask about a US public company, and stop recording. "
         "You'll hear a short spoken summary; the full answer with sources appears below it.")
with st.expander("Example questions"):
    for example in EXAMPLES:
        st.markdown(f"- {example}")

missing = [label for label, value in (("Cartesia key", cartesia_key), ("OpenRouter key", openrouter_key), ("SEC email", sec_email)) if not value.strip()]
if missing:
    st.info("To start, add in the sidebar: " + ", ".join(missing) + ".", icon=":material/key:")

recording = st.audio_input("Record your question", disabled=bool(missing))
typed = st.chat_input("…or type a question", disabled=bool(missing))

question, timings = None, {}
if recording is not None and st.session_state.get("last_recording") != recording.file_id:
    st.session_state["last_recording"] = recording.file_id
    started = time.perf_counter()
    try:
        with st.spinner("Listening back…"):
            question = transcribe(recording.getvalue(), cartesia_key, language, filename=recording.name or "question.wav")
        timings["Speech to text"] = time.perf_counter() - started
    except VoiceError as error:
        st.error(str(error), icon=":material/mic_off:")
elif typed:
    question = typed

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            user_agent = validate_sec_user_agent(sec_email)
            tools = build_tools(user_agent, None)  # filing search needs the vector index from the main app; SEC figures work without it
            started = time.perf_counter()
            with st.spinner("Researching…"):
                answer, messages = run_research_agent(question, openrouter_key, model, tools, provider="openrouter", language=language)
            timings["Research"] = time.perf_counter() - started
        except PromptInjectionError as error:
            st.warning(str(error), icon=":material/shield:")
            st.stop()
        except Exception as error:  # demo page: show any failure plainly instead of a traceback
            st.error(f"The assistant couldn't answer: {error}", icon=":material/error:")
            st.stop()

        summary = spoken_summary(answer)
        if speak_answers and summary:
            try:
                started = time.perf_counter()
                with st.spinner("Speaking…"):
                    audio = synthesize(summary, cartesia_key, language)
                timings["Text to speech"] = time.perf_counter() - started
                st.audio(audio, format="audio/wav", autoplay=True)
                st.caption(f"Spoken: “{summary}”")
            except VoiceError as error:
                st.warning(f"{error} Showing the answer as text only.", icon=":material/volume_off:")

        st.markdown(answer)
        citations = extract_citations(_tool_results(messages))
        if citations:
            st.markdown("**Sources**\n\n" + format_citations_markdown(citations))
        if timings:
            st.caption(" · ".join(f"{name} {seconds:.1f} s" for name, seconds in timings.items()))

st.divider()
st.caption("Demo only. Market Lens explains evidence from SEC filings; it does not give investment advice.")
