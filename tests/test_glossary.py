"""Newcomer help: glossary search, term lookup in the help bot, and the welcome card."""

from __future__ import annotations

from finance_research.conversation import answer_help_question
from finance_research.glossary import GLOSSARY, TAB_TOUR, all_terms, lookup_term, search_glossary


def test_every_entry_is_complete() -> None:
    for entry in all_terms():
        assert entry["term"] and len(entry["meaning"]) > 20 and len(entry["why"]) > 10, entry
    assert len(all_terms()) == sum(len(entries) for entries in GLOSSARY.values())
    assert len({entry["term"] for entry in all_terms()}) == len(all_terms())  # no duplicates


def test_tab_tour_matches_the_app_tabs() -> None:
    from finance_research.i18n import STRINGS

    tab_names = {value["en"] for key, value in STRINGS.items() if key.startswith("tab_")}
    assert {tab for tab, _shows, _use in TAB_TOUR} == tab_names


def test_search_puts_term_matches_first() -> None:
    results = search_glossary("margin")
    assert results[0]["term"] in {"Operating margin", "Net margin"}
    assert search_glossary("10k")[0]["term"] == "10-K (annual report)"
    assert search_glossary("zzzz") == []
    assert len(search_glossary("")) == len(all_terms())


def test_lookup_prefers_the_longest_name() -> None:
    assert lookup_term("What does net margin mean?")["term"] == "Net margin"
    assert lookup_term("what is an 8-K")["term"] == "8-K (major event report)"
    assert lookup_term("Explain EPS please")["term"] == "EPS (earnings per share)"
    assert lookup_term("hello there") is None


def test_help_bot_defines_terms_but_keeps_how_to_answers() -> None:
    reply = answer_help_question("What is a 10-K?")
    assert reply["title"] == "10-K (annual report)" and "Why it matters" in reply["answer"]
    assert answer_help_question("What is RAG?")["title"].startswith("RAG")
    assert answer_help_question("How do I index filings?")["title"] != "Filing"  # still the how-to topic


def test_welcome_card_and_guide_tab_render(monkeypatch) -> None:
    from test_shared_key import _start_as_guest  # pytest puts the tests folder on the import path

    at, _ = _start_as_guest(monkeypatch)
    assert any("New here?" in block.value for block in at.markdown)
    assert any("AI key" in block.value and "shared key" in block.value for block in at.markdown)
    at.text_input(key="glossary_query").set_value("dividend").run()
    assert not at.exception, at.exception
    assert any("Dividend yield" in block.value for block in at.markdown)
    at.button(key="welcome-hide").click().run()
    assert not any("New here?" in block.value for block in at.markdown)