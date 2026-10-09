import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from churn import config  # noqa: E402
from churn.analysis import STRATEGIES, STRATEGY_CUSTOM  # noqa: E402

APP = str(config.PROJECT_ROOT / "app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception
    return at


def _pages(app):
    return app.sidebar.radio(key="page").options


def _visit_all_pages(app, label):
    for page in _pages(app):
        app.sidebar.radio(key="page").set_value(page).run()
        assert not app.exception, f"{page} / {label}: {app.exception}"
        assert app.title[0].value in page


def test_nine_sections_are_available(app):
    assert len(_pages(app)) == 9
    assert _pages(app)[0].startswith("0. Executive Briefing")  # landing page
    assert "Hyperparameter Tuning" in _pages(app)[4]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_page_renders_under_every_strategy(app, strategy):
    app.sidebar.radio(key="strategy").set_value(strategy).run()
    _visit_all_pages(app, strategy)


def test_custom_threshold_slider_changes_thresholds(app):
    app.sidebar.radio(key="page").set_value(_pages(app)[3]).run()
    app.sidebar.radio(key="strategy").set_value(STRATEGY_CUSTOM).run()
    slider = app.sidebar.slider(key="custom_t")
    assert not slider.disabled
    assert (slider.min, slider.max) == (0.10, 0.90)
    slider.set_value(0.3).run()
    assert not app.exception
    assert "t = 0.300" in " ".join(m.value for m in app.sidebar.markdown)


def test_balanced_class_weighting_renders(app):
    weighting = app.sidebar.radio(key="weighting")
    weighting.set_value(weighting.options[1]).run()
    assert not app.exception
    _visit_all_pages(app, "balanced weighting")
    weighting.set_value(weighting.options[0]).run()


def test_prediction_tool_reacts_to_inputs(app):
    app.sidebar.radio(key="page").set_value(_pages(app)[7]).run()
    probability = lambda: next(m.value for m in app.metric if m.label == "Churn probability")
    typical = probability()
    app.number_input(key="in_Age").set_value(60)
    app.selectbox(key="in_Geography").set_value("Germany")
    app.toggle(key="in_IsActiveMember").set_value(False).run()
    assert not app.exception
    risky = probability()
    assert float(risky.rstrip("%")) > float(typical.rstrip("%"))


def test_every_metric_column_and_control_has_a_tooltip(app):
    app.sidebar.radio(key="page").set_value(_pages(app)[3]).run()
    for key in ("page", "weighting", "strategy"):
        assert app.sidebar.radio(key=key).help
    assert app.sidebar.slider(key="custom_t").help
    assert any("Key terms" in e.label for e in app.expander)
    assert any("Glossary" in e.label for e in app.sidebar.expander)


@pytest.mark.skipif(not (config.OUTPUT_DIR / "tuning_cv_folds.csv").is_file(), reason="run main.py first")
def test_tuning_page_matches_the_saved_report(app):
    """The dashboard recomputes the before/after numbers; they must equal the ones main.py wrote."""
    import pandas as pd

    app.sidebar.radio(key="page").set_value(_pages(app)[4]).run()
    assert not app.exception
    labels = {m.label: m.value for m in app.metric}
    saved = pd.read_csv(config.OUTPUT_DIR / "tuning_comparison.csv")
    for name in ("Random Forest", "XGBoost"):
        for stage, label in [("CV mean", "CV ROC-AUC"), ("Test", "Test ROC-AUC")]:
            row = saved[(saved["Model"] == name) & (saved["Stage"] == stage) & (saved["Metric"] == "ROC_AUC")].iloc[0]
            assert labels[f"{name} · {label}"] == f"{row['Tuned']:.3f}"
    assert "Best model overall" in " ".join(m.value for m in app.markdown)


def test_briefing_answers_the_three_questions_and_journey_navigates(app):
    app.sidebar.radio(key="page").set_value(_pages(app)[0]).run()
    assert not app.exception
    html = " ".join(m.value for m in app.markdown)
    for question in ("The mission", "The investigation", "Success criteria"):
        assert question in html
    stages = app.get("button_group")[0] if app.get("button_group") else None
    assert stages is not None and len(stages.options) == 6
    stages.set_value("Tuning").run()
    assert not app.exception
    button = next(b for b in app.button if b.key.startswith("journey_go_"))
    assert button.key == "journey_go_4"
    button.click().run()
    assert app.sidebar.radio(key="page").value == _pages(app)[4]
    assert "Hyperparameter Tuning" in app.title[0].value
