"""English and Hebrew interface text: complete, consistent, and formatted with the same placeholders."""

import re
from string import Formatter

import pytest

from churn.glossary import (
    GLOSSARY,
    GLOSSARY_HE,
    HYPERPARAMETERS,
    HYPERPARAMETERS_HE,
    METRIC_HELP,
    METRIC_HELP_HE,
    TERM_NAMES_HE,
    define,
    term_name,
)
from churn.i18n import LANGUAGES, PAGE_SUBTITLES_HE, PAGE_TITLES, STRINGS, page_labels, tr
from churn.story_game import INCENTIVE_META, INCENTIVE_META_HE, PERSONA_HE, PERSONA_LINES, PERSONA_LINES_HE

HEBREW = re.compile(r"[֐-׿]")


def fields(text: str) -> set[str]:
    return {name for _, name, _, _ in Formatter().parse(text) if name}


def test_every_string_exists_in_both_languages_with_the_same_placeholders():
    assert set(LANGUAGES) == {"en", "he"}
    for key, entry in STRINGS.items():
        assert set(entry) == {"en", "he"}, key
        en, he = entry["en"], entry["he"]
        assert type(en) is type(he), key
        if isinstance(en, str):
            assert fields(en) == fields(he), key
            if en:
                assert HEBREW.search(he) or key in {"language"} or not HEBREW.search(en), key
        else:  # lists of stages / stage details
            assert len(en) == len(he), key


def test_hebrew_text_is_actually_hebrew():
    for key in ("navigate", "model_config", "hero_title", "triggers", "journey", "key_terms", "glossary"):
        assert HEBREW.search(tr(key, "he")), key
    assert all(HEBREW.search(t) for t in PAGE_TITLES["he"])


def test_page_titles_and_subtitles():
    assert len(PAGE_TITLES["en"]) == len(PAGE_TITLES["he"]) == len(PAGE_SUBTITLES_HE) == 10
    assert page_labels("en")[3] == "3. Model Explorer & Comparison"
    assert page_labels("he")[0].startswith("0. ")
    assert PAGE_SUBTITLES_HE[3].format(n_models=5, n_test="2,000").startswith("5 מודלים")


def test_tr_formats_and_falls_back():
    assert tr("hero_title", "en", rate="20.4%").startswith("20.4% of customers")
    assert "20.4%" in tr("hero_title", "he", rate="20.4%")
    assert tr("navigate", "fr") == "Navigate"  # unknown language -> English
    with pytest.raises(KeyError):
        tr("no_such_key", "en")


def test_glossary_is_complete_in_hebrew():
    assert set(GLOSSARY_HE) == set(GLOSSARY) == set(TERM_NAMES_HE)
    assert set(HYPERPARAMETERS_HE) == set(HYPERPARAMETERS)
    assert set(METRIC_HELP_HE) == set(METRIC_HELP)
    for term in GLOSSARY:
        assert HEBREW.search(define(term, "he")) and define(term, "en") == GLOSSARY[term]
    assert term_name("ROC_AUC", "en") == "ROC-AUC" and "Recall" in term_name("Recall", "he")
    assert all(len(v) < 300 for v in {**GLOSSARY_HE, **HYPERPARAMETERS_HE}.values())  # tooltip-sized


def test_game_personas_and_incentives_have_hebrew_text():
    assert set(PERSONA_HE) == set(PERSONA_LINES) == set(PERSONA_LINES_HE)
    for key, lines in PERSONA_LINES.items():
        assert set(PERSONA_LINES_HE[key]) == set(lines), key
        assert all(HEBREW.search(v) for v in PERSONA_LINES_HE[key].values()), key
    assert set(INCENTIVE_META_HE) == {meta[0] for meta in INCENTIVE_META.values()}
