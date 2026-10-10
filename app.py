"""Customer Churn Analytics: a Streamlit data science dashboard.

Run with:
    streamlit run app.py

All numbers come from the project's own package (churn/). Models, cross-
validation and the optimised thresholds use the training set only; the test
set is used only to report performance.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

from churn import config
from churn.animation import GIF_PATH, make_boosting_gif
from churn.analysis import (
    STRATEGIES,
    STRATEGY_CUSTOM,
    STRATEGY_DEFAULT,
    STRATEGY_OPTIMIZED,
    Analysis,
    compute_permutation_importance,
    run_analysis,
)
from churn.evaluation import build_comparison_table
from churn.explain import explain_prediction
from churn.formatting import fmt3, fmt3_signed
from churn.story_game import assets_fingerprint, build_payload, render_html
from churn.glossary import GLOSSARY, HYPERPARAMETERS, METRIC_HELP, define
from churn.insights import (
    GAP_STD_RULE,
    MIN_SEGMENT_N,
    analytical_summary,
    best_model,
    format_segments,
    generalisation_notes,
    generalisation_table,
    importance_insights,
    paired_fold_comparison,
    segment_stats,
)
from churn.models import (
    CATBOOST_PARAMS,
    FOREST_PARAMS,
    LIGHTGBM_PARAMS,
    LOGISTIC_PARAMS,
    TREE_MODELS,
    XGBOOST_PARAMS,
    balanced_pos_weight,
    positive_class_proba,
    predict_models,
)
from churn.thresholds import apply_threshold, select_cost_threshold, threshold_curve
from churn.tuning import (
    PARAM_GRIDS,
    TUNED_MODELS,
    candidates_table,
    comparison_table,
    load_tuning_results,
    n_candidates,
    tuned_name,
    tuning_insights,
)
from churn.validation import CV_METRICS, format_mean_std, summarize_cv

# --- Constants ------------------------------------------------------------------
PAGES = [
    "0. Executive Briefing & Journey Map",
    "1. Project Overview & Data Inspection",
    "2. Preprocessing & Pipeline Architecture",
    "3. Model Explorer & Comparison",
    "4. Hyperparameter Tuning: Before vs After",
    "5. Cross-Validation & Robustness",
    "6. Feature Importance",
    "7. Interactive Churn Prediction Tool",
    "8. Methodology & Future Roadmap",
    "9. The Chief Retention Officer: A Machine Learning Decision Story",
]
# Reference categorical palette, slots 1-5 in fixed order (validated together for colour-vision deficiency).
MODEL_COLORS = {"Logistic Regression": "#2a78d6", "Random Forest": "#eb6834", "XGBoost": "#1baf7a",
                "LightGBM": "#eda100", "CatBoost": "#e87ba4"}
PILL_CLASS = {"Logistic Regression": "blue", "Random Forest": "orange", "XGBoost": "aqua",
              "LightGBM": "yellow", "CatBoost": "magenta"}
IMPORTANCE_METHOD = {"Random Forest": "impurity decrease", "XGBoost": "gain", "LightGBM": "gain",
                     "CatBoost": "prediction change"}
CLASS_COLORS = {"Stayed": "#8a8984", "Exited": "#9085e9"}
RAISES, LOWERS = "#e34948", "#008300"
BASELINE = "#8a8984"
CHART_CONFIG = {"displayModeBar": False}
METRIC_COLS = ["Precision", "Recall", "F1", "Accuracy", "ROC_AUC"]
WEIGHTING = {"None (equal weights)": None, "Balanced (up-weight churners)": "balanced"}
PERMUTATION_REPEATS = 10

FEATURE_INFO = {
    "CustomerId": ("Identifier (excluded)", "Unique customer number. Excluded: no general signal, allows memorisation."),
    "Surname": ("Identifier (excluded)", "Customer surname. Excluded: an identifier, and a fairness risk."),
    "CreditScore": ("Numeric", "Credit score of the customer."),
    "Geography": ("Categorical", "Country of residence."),
    "Gender": ("Categorical", "Customer gender."),
    "Age": ("Numeric", "Age in years."),
    "Tenure": ("Numeric", "Years as a customer of the bank."),
    "Balance": ("Numeric", "Account balance (currency not documented in the dataset)."),
    "NumOfProducts": ("Numeric", "Number of bank products held."),
    "HasCrCard": ("Binary (numeric)", "1 if the customer holds a credit card."),
    "IsActiveMember": ("Binary (numeric)", "1 if the customer is classed as active."),
    "EstimatedSalary": ("Numeric", "Estimated annual salary."),
    "Exited": ("Target", "1 if the customer left the bank, 0 if they stayed."),
}
FEATURE_LABELS = {
    "CreditScore": "Credit score", "Age": "Age", "Tenure": "Tenure (years)", "Balance": "Balance",
    "NumOfProducts": "Number of products", "HasCrCard": "Has credit card", "IsActiveMember": "Active member",
    "EstimatedSalary": "Estimated salary", "Geography": "Geography", "Gender": "Gender",
}

st.set_page_config(
    page_title="Customer Churn Analytics", page_icon="📊", layout="wide", initial_sidebar_state="auto"
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]) { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 3.2rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.page-kicker { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.6; margin-bottom: -0.4rem; }
.page-sub { font-size: 1.02rem; opacity: 0.78; margin-top: -0.3rem; margin-bottom: 1.2rem; }
.wwh h4 { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.65; margin: 0 0 0.35rem 0; }
.wwh p { margin: 0; font-size: 0.94rem; line-height: 1.5; }
.pill { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
        border: 1px solid rgba(128,128,128,0.35); margin-right: 0.35rem; }
.pill.blue { background: rgba(42,120,214,0.14); border-color: rgba(42,120,214,0.45); }
.pill.orange { background: rgba(235,104,52,0.14); border-color: rgba(235,104,52,0.45); }
.pill.aqua { background: rgba(27,175,122,0.14); border-color: rgba(27,175,122,0.45); }
.pill.yellow { background: rgba(237,161,0,0.14); border-color: rgba(237,161,0,0.5); }
.pill.magenta { background: rgba(232,123,164,0.16); border-color: rgba(232,123,164,0.5); }
.verdict { border-radius: 12px; padding: 1rem 1.2rem; border: 1px solid rgba(128,128,128,0.25); border-left-width: 6px; margin-bottom: 0.8rem; }
.verdict.high { border-left-color: #e34948; background: rgba(227,73,72,0.08); }
.verdict.elevated { border-left-color: #eda100; background: rgba(237,161,0,0.09); }
.verdict.low { border-left-color: #1baf7a; background: rgba(27,175,122,0.08); }
.verdict.best { border-left-color: #2a78d6; background: rgba(42,120,214,0.07); }
.verdict h3 { margin: 0 0 0.25rem 0; font-size: 1.3rem; }
.verdict p { margin: 0; opacity: 0.88; }
section[data-testid="stSidebar"] .stRadio label p { font-size: 0.93rem; }
.hero { border-radius: 18px; padding: 1.6rem 1.8rem 1.4rem 1.8rem; margin-bottom: 1.1rem;
        background: linear-gradient(135deg, rgba(42,120,214,0.16) 0%, rgba(144,133,233,0.12) 55%, rgba(27,175,122,0.10) 100%);
        border: 1px solid rgba(42,120,214,0.28); }
.hero .eyebrow { font-size: 0.76rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.7; }
.hero h2 { margin: 0.25rem 0 0.45rem 0; font-size: 1.85rem; line-height: 1.2; }
.hero p { margin: 0; font-size: 1.03rem; opacity: 0.85; max-width: 60rem; }
.mission { border-radius: 14px; padding: 1rem 1.1rem; height: 100%; border: 1px solid rgba(128,128,128,0.25);
           border-top-width: 5px; }
.mission.m1 { border-top-color: #2a78d6; } .mission.m2 { border-top-color: #eb6834; } .mission.m3 { border-top-color: #1baf7a; }
.mission .tag { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.65; }
.mission h4 { margin: 0.2rem 0 0.45rem 0; font-size: 1.08rem; }
.mission p, .mission li { font-size: 0.92rem; line-height: 1.5; margin: 0; }
.mission ul { margin: 0.35rem 0 0 1.1rem; padding: 0; }
.trigger { border-radius: 12px; padding: 0.8rem 1rem; border: 1px solid rgba(128,128,128,0.25); height: 100%; }
.trigger .big { font-size: 1.6rem; font-weight: 700; line-height: 1.1; }
.trigger .lbl { font-size: 0.82rem; opacity: 0.75; }
/* Responsive: tablets and phones (Streamlit already stacks st.columns below ~640px) */
.hero, .mission, .trigger, .verdict, .wwh { max-width: 100%; overflow-wrap: anywhere; }
@media (max-width: 900px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .hero { padding: 1.2rem 1.2rem 1.1rem 1.2rem; }
  .hero h2 { font-size: 1.5rem; }
}
@media (max-width: 640px) {
  .block-container { padding-top: 2.6rem; padding-left: 0.75rem; padding-right: 0.75rem; }
  h1 { font-size: 1.75rem !important; }
  .page-sub { font-size: 0.95rem; }
  .hero { padding: 1rem; border-radius: 14px; }
  .hero h2 { font-size: 1.3rem; }
  .hero p { font-size: 0.95rem; }
  .mission, .trigger { margin-bottom: 0.6rem; }
  .verdict { padding: 0.8rem 0.9rem; }
  .verdict h3 { font-size: 1.1rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# --- Data (cached per class-weighting setting) -----------------------------------------
@st.cache_resource(show_spinner="Running 5-fold cross-validation and training 5 models on the training set (≈1 min, once per setting)…")
def get_analysis(class_weight: str | None) -> Analysis:
    return run_analysis(class_weight=class_weight, permutation_repeats=None)


@st.cache_resource(show_spinner="Computing permutation importance on the test set (≈20 s, once per setting)…")
def get_permutation(class_weight: str | None) -> dict[str, pd.DataFrame]:
    a = get_analysis(class_weight)
    return compute_permutation_importance(a.models, a.X_test, a.y_test, n_repeats=PERMUTATION_REPEATS)


@st.cache_resource(show_spinner="Loading the tuned models saved by main.py…")
def get_tuning(_stamp: float) -> dict | None:
    """Tuning results written by main.py. _stamp (file modification time) reloads them after a new run."""
    return load_tuning_results()


def tuning_stamp() -> float:
    path = config.OUTPUT_DIR / "tuning_cv_folds.csv"
    return path.stat().st_mtime if path.is_file() else 0.0


@st.cache_resource(show_spinner="Rendering the gradient-boosting animation (once)…")
def boosting_gif() -> Path:
    """The committed GIF in assets/, regenerated only if it is missing."""
    return GIF_PATH if GIF_PATH.is_file() else make_boosting_gif()


# --- Shared UI helpers ---------------------------------------------------------------
def page_header(index: int, subtitle: str) -> None:
    number, title = PAGES[index].split(". ", 1)
    st.markdown(f'<div class="page-kicker">Section {number}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def what_why_how(what: str, why: str, how: str) -> None:
    """The three-part explanation used in every section."""
    cols = st.columns(3)
    for col, (label, text) in zip(cols, [("What we did", what), ("Why it matters", why), ("How to read it", how)]):
        with col.container(border=True):
            st.markdown(f'<div class="wwh"><h4>{label}</h4><p>{text}</p></div>', unsafe_allow_html=True)


def plotly(fig: go.Figure, height: int = 380) -> None:
    fig.update_layout(height=height, margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)


# All metric values are displayed through churn.formatting (fmt3 / fmt3_signed), the same rule used by the
# CSV files, the CLI and the report. Plotly hover labels receive pre-formatted strings (customdata), because
# the browser's own number formatting rounds ties differently.


def hover3(*columns) -> np.ndarray:
    """Pre-formatted hover strings: one column per series, one row per point."""
    return np.column_stack([[fmt3(v) for v in col] for col in columns])


def styled_metrics(table: pd.DataFrame, axis: int = 0):
    """Raw values decide the bold 'best' cell; display uses the shared three-decimal rule."""
    return table.style.format(fmt3).highlight_max(axis=axis, props="font-weight: 700;")


def column_help(columns) -> dict:
    """Header tooltips for every metric column in a table (hover the column name)."""
    config_ = {}
    for col in columns:
        base = str(col).split(" ")[0].replace("-", "_")
        if base in METRIC_HELP:
            config_[col] = st.column_config.Column(help=METRIC_HELP[base])
        elif col in ("Threshold", "Model", "Hyperparameter"):
            config_[col] = st.column_config.Column(help=define({"Threshold": "Threshold", "Model": "Baseline",
                                                                  "Hyperparameter": "Hyperparameter"}[col]))
    return config_


def metrics_dataframe(table: pd.DataFrame, axis: int = 0) -> None:
    """A metrics table with the shared rounding, bold best values and a tooltip on every metric header."""
    st.dataframe(styled_metrics(table, axis), width="stretch", column_config=column_help(table.columns))


def key_terms(*terms: str) -> None:
    """A collapsible panel defining the statistical terms used in a section."""
    with st.expander("ℹ️ Key terms in this section"):
        st.markdown("\n".join(f"- **{t.replace('_', '-')}:** {define(t)}" for t in terms))


def display_value(col: str, value) -> str:
    if col in ("HasCrCard", "IsActiveMember"):
        return "Yes" if int(value) else "No"
    if isinstance(value, str):
        return value
    return f"{value:,.2f}" if col in ("Balance", "EstimatedSalary") else f"{value:,}"


def bullets(lines: list[str]) -> None:
    st.markdown("\n".join(f"- {line}" for line in lines))


def confusion_figure(cm: np.ndarray, title: str) -> go.Figure:
    labels = ["Stayed (0)", "Exited (1)"]
    shares = cm / cm.sum(axis=1, keepdims=True)
    names = [["True negative", "False positive"], ["False negative", "True positive"]]
    text = [[f"<b>{cm[i, j]:,}</b><br>{shares[i, j]:.1%} of actual<br>{names[i][j]}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels, text=text, texttemplate="%{text}", textfont=dict(size=12),
        colorscale="Blues", showscale=False, xgap=3, ygap=3,
        hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z} customers<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)),
        xaxis=dict(title="Predicted class"), yaxis=dict(title="Actual class", autorange="reversed"),
    )
    return fig


# --- Sidebar ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 Churn Analytics")
    st.caption("Bank customer churn · Logistic Regression · Random Forest · XGBoost · LightGBM · CatBoost")
    page = st.radio("Navigate", PAGES, key="page", help="Each section explains what was done, why, and how to read it.")
    st.divider()
    st.markdown("**Model configuration**")
    weighting_label = st.radio(
        "Class balancing / weighting", list(WEIGHTING), key="weighting",
        help=(
            "Balanced: errors on churners count more during training (inverse class frequency). Logistic "
            "Regression, Random Forest and LightGBM use class_weight='balanced'; XGBoost uses scale_pos_weight = "
            "#stayed / #churned from the training labels; CatBoost uses auto_class_weights='Balanced'. Models are "
            "retrained the first time you switch."
        ),
    )
    class_weight = WEIGHTING[weighting_label]
    strategy = st.radio(
        "Threshold strategy", STRATEGIES, key="strategy",
        help=(
            "Default: P(churn) ≥ 0.5. CV-optimized: per-model threshold that maximised F1 in 5-fold "
            "cross-validation on the training set (no test data used). Custom: use the slider."
        ),
    )
    custom_t = st.slider(
        "Decision threshold (custom)", 0.10, 0.90, 0.50, 0.01, key="custom_t",
        disabled=strategy != STRATEGY_CUSTOM,
        help="Active only with the Custom strategy. A customer is flagged when P(churn) ≥ threshold. "
             "Lower threshold → more churners caught (Recall ↑) but more false alarms (Precision ↓).",
    )

try:
    A = get_analysis(class_weight)
except (FileNotFoundError, ValueError) as error:
    st.error(f"The dataset could not be loaded or validated: {error}")
    st.stop()

EVALS = A.evaluate_strategy(strategy, custom_t)
TABLE = Analysis.metrics_table(EVALS)
THRESHOLDS = {name: e.threshold for name, e in EVALS.items()}

with st.sidebar:
    for name in A.models:
        st.markdown(f'<span class="pill {PILL_CLASS[name]}">{name}</span><b>t = {fmt3(THRESHOLDS[name])}</b>',
                    unsafe_allow_html=True)
    if strategy == STRATEGY_CUSTOM:
        st.warning("Exploratory mode: choosing a threshold by watching test-set metrics is test-set tuning. "
                   "Use the CV-optimized strategy for an honest estimate.", icon="⚠️")
    st.divider()
    st.caption(
        f"Dataset: `{config.DATA_PATH.name}` · {len(A.df):,} customers · train {len(A.X_train):,} / "
        f"test {len(A.X_test):,} · {config.CV_FOLDS}-fold CV · seed {config.RANDOM_STATE}"
    )
    st.caption("Educational project. Not validated for production banking decisions.")
    with st.expander("📖 Glossary of terms"):
        st.markdown("\n".join(f"- **{t.replace('_', '-')}:** {d}" for t, d in GLOSSARY.items()))

CONFIG_LINE = (
    f"Configuration: weighting **{weighting_label.split(' (')[0]}** · strategy **{strategy}** · thresholds "
    + " · ".join(f"{m} **{fmt3(t)}**" for m, t in THRESHOLDS.items())
)


# =============================================================================
# 0. Executive briefing & analytical journey map
# =============================================================================
def _trigger(feature: str) -> dict:
    """Highest- and lowest-churn levels of a feature (training set, segments with n >= MIN_SEGMENT_N)."""
    stats = segment_stats(A.X_train, A.y_train, feature)
    stats = stats[~stats["small"]]
    hi, lo = stats["rate"].idxmax(), stats["rate"].idxmin()
    return {"hi": hi, "lo": lo, "hi_rate": stats.loc[hi, "rate"], "lo_rate": stats.loc[lo, "rate"],
            "hi_n": int(stats.loc[hi, "n"]), "ratio": stats.loc[hi, "rate"] / stats.loc[lo, "rate"]}


def _go_to(target: str) -> None:
    st.session_state["page"] = target


JOURNEY = [
    ("Data Inspection", "🔍", 1), ("Preprocessing", "🧱", 2), ("Base Models", "📐", 3),
    ("Advanced Boosters", "🚀", 3), ("Tuning", "🎛️", 4), ("Final Deployment", "🎯", 7),
]


def page_briefing() -> None:
    page_header(0, "Start here: what this project set out to do, what it found, and how to navigate the evidence.")
    cv = A.cv_summary()
    best = best_model(cv)
    default, tuned = A.evaluate_strategy(STRATEGY_DEFAULT)[best], A.evaluate_strategy(STRATEGY_OPTIMIZED)[best]
    churned = int(A.df[config.TARGET].sum())

    st.markdown(
        f"""<div class="hero"><div class="eyebrow">Executive briefing · bank customer churn</div>
<h2>{A.df[config.TARGET].mean():.1%} of customers left the bank. Can we see it coming?</h2>
<p>{churned:,} of {len(A.df):,} customers exited. This project builds and stress-tests {len(A.models)} machine-learning
models that rank every customer by churn risk, so a retention team can act <i>before</i> revenue walks out of the door.</p></div>""",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3, gap="medium")
    m1.markdown(
        """<div class="mission m1"><div class="tag">The mission · where we are diving</div>
<h4>Understand and predict churn</h4>
<p>Every customer who leaves takes their balances, products and future revenue with them. Retaining a customer is
typically cheaper than acquiring a new one, so knowing <b>who is likely to leave</b> turns churn from a loss we
report into a risk we can manage.</p></div>""", unsafe_allow_html=True)
    m2.markdown(
        """<div class="mission m2"><div class="tag">The investigation · what we look for</div>
<h4>Behavioural and demographic triggers</h4>
<p>Which customer signals separate leavers from stayers?</p>
<ul><li>Number of products held</li><li>Activity status</li><li>Age and geography</li>
<li>Balance patterns</li></ul></div>""", unsafe_allow_html=True)
    m3.markdown(
        """<div class="mission m3"><div class="tag">Success criteria · how we will know</div>
<h4>Rigour, not flattering accuracy</h4>
<p>~80% accuracy is free (predict "stays" for everyone), so it proves nothing. Instead:</p>
<ul><li><b>ROC-AUC</b>: ranking quality, threshold-free</li><li><b>5-fold cross-validation</b>: mean ± std, not one lucky split</li>
<li><b>Threshold tuning</b> on training data to trade Precision for Recall</li></ul></div>""", unsafe_allow_html=True)

    st.subheader("Headline results")
    k = st.columns(4)
    k[0].metric("Selected model", best, border=True,
                help="Highest mean ROC-AUC in 5-fold cross-validation on the training set (the test set only confirms).")
    k[1].metric("CV ROC-AUC (mean ± std)", f"{fmt3(cv.loc[best, 'ROC_AUC mean'])} ± {fmt3(cv.loc[best, 'ROC_AUC std'])}",
                border=True, help=f"Mean ± std over {config.CV_FOLDS} folds. " + define("ROC_AUC"))
    k[2].metric("Churners caught (Recall)", f"{tuned.metrics['Recall']:.0%}", border=True,
                delta=f"{(tuned.metrics['Recall'] - default.metrics['Recall']) * 100:+.0f} pp vs 0.5 cutoff",
                help=f"Test-set Recall at the CV-optimized threshold ({fmt3(tuned.threshold)}) vs the default 0.5. "
                     + define("Recall"))
    k[3].metric("Flags that are real churners", f"{tuned.metrics['Precision']:.0%}", border=True,
                delta=f"{(tuned.metrics['Precision'] - default.metrics['Precision']) * 100:+.0f} pp vs 0.5 cutoff",
                delta_color="off",
                help="Test-set Precision at the CV-optimized threshold: the price of catching more churners. "
                     + define("Precision"))
    st.caption(f"Live values for the current sidebar configuration (weighting: {weighting_label.split(' (')[0]}). "
               "Recall and Precision are measured on the untouched test set.")

    st.subheader("What the data says: churn triggers")
    st.markdown("Churn rate by segment in the **training set** (segments with fewer than "
                f"{MIN_SEGMENT_N} customers excluded). These are associations, not proven causes.")
    labels = {"NumOfProducts": ("Products held", lambda v: f"{v} product{'s' if v != 1 else ''}"),
              "IsActiveMember": ("Activity status", lambda v: "Inactive" if v == 0 else "Active"),
              "Age": ("Age band", lambda v: f"Age {v}"), "Geography": ("Geography", str)}
    cols = st.columns(len(labels), gap="small")
    for col, (feature, (title, fmt)) in zip(cols, labels.items()):
        t = _trigger(feature)
        col.markdown(
            f"""<div class="trigger"><div class="lbl">{title}</div><div class="big">{t['hi_rate']:.0%}</div>
<div class="lbl">churn for <b>{fmt(t['hi'])}</b> (n = {t['hi_n']:,})<br>vs {t['lo_rate']:.0%} for {fmt(t['lo'])}
· <b>{t['ratio']:.1f}×</b></div></div>""", unsafe_allow_html=True)

    st.subheader("Analytical journey map")
    st.markdown("Six stages from raw data to a decision tool. Select a stage to see what was done and what was "
                "measured, then jump to the full evidence.")
    names = [name for name, _, _ in JOURNEY]
    stage = st.segmented_control("Stage", names, default=names[0], key="journey_stage",
                                 help="Each stage links to the dashboard section with the full analysis.",
                                 label_visibility="collapsed") or names[0]
    current = names.index(stage)

    fig = go.Figure()
    fig.add_scatter(x=list(range(len(JOURNEY))), y=[0] * len(JOURNEY), mode="lines",
                    line=dict(color=BASELINE, width=3), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=list(range(current + 1)), y=[0] * (current + 1), mode="lines",
                    line=dict(color="#2a78d6", width=6), hoverinfo="skip", showlegend=False)
    fig.add_scatter(
        x=list(range(len(JOURNEY))), y=[0] * len(JOURNEY), mode="markers+text",
        marker=dict(size=[46 if i == current else 34 for i in range(len(JOURNEY))],
                    color=["#2a78d6" if i <= current else "rgba(138,137,132,0.25)" for i in range(len(JOURNEY))],
                    line=dict(width=3, color="white")),
        text=[icon for _, icon, _ in JOURNEY], textfont=dict(size=18), textposition="middle center",
        hovertext=[f"Step {i + 1}: {n}" for i, n in enumerate(names)], hoverinfo="text", showlegend=False,
    )
    for i, n in enumerate(names):
        label = f"{i + 1}<br>{n.replace(' ', '<br>', 1)}"  # two short lines, so labels fit on phones too
        fig.add_annotation(x=i, y=-0.68, text=f"<b>{label}</b>" if i == current else label,
                           showarrow=False, font=dict(size=12))
    fig.update_layout(xaxis=dict(visible=False, range=[-0.5, len(JOURNEY) - 0.5]),
                      yaxis=dict(visible=False, range=[-1.15, 0.5]), plot_bgcolor="rgba(0,0,0,0)")
    plotly(fig, 190)

    lr, rf = cv.loc["Logistic Regression", "ROC_AUC mean"], cv.loc["Random Forest", "ROC_AUC mean"]
    boosters = [m for m in ["XGBoost", "LightGBM", "CatBoost"] if m in cv.index]
    T = get_tuning(tuning_stamp())
    if T is not None:
        rf_tuned = summarize_cv(T["tuned_folds"]).loc["Random Forest", "ROC_AUC mean"]
        tuning_result = (f"Random Forest CV ROC-AUC {fmt3(rf)} → {fmt3(rf_tuned)} after GridSearchCV; XGBoost's "
                         "defaults were already near-optimal.")
    else:
        tuning_result = "Run `python main.py` to produce the tuning results."
    split = (A.y_train.mean(), A.y_test.mean())
    details = [
        ("Validated 10,000 customers × 13 columns: no missing values, no duplicates, binary target. Found the one "
         "material issue: class imbalance.", f"Churn rate {A.base_rate:.1%}: accuracy alone is misleading."),
        ("Dropped identifiers, one-hot encoded Geography and Gender, scaled numerics for the linear model, and put "
         "everything inside a scikit-learn Pipeline after a stratified 80/20 split.",
         f"Train churn {split[0]:.1%} vs test {split[1]:.1%}; preprocessing learns from training rows only (no leakage)."),
        ("Established reference points: an interpretable linear model and a bagging ensemble of deep trees.",
         f"CV ROC-AUC: Logistic Regression {fmt3(lr)}, Random Forest {fmt3(rf)}. Non-linear patterns matter."),
        ("Added three gradient-boosting implementations that build shallow trees sequentially, each correcting "
         "the errors of the last.", "CV ROC-AUC: " + ", ".join(f"{m} {fmt3(cv.loc[m, 'ROC_AUC mean'])}" for m in boosters)
         + ". The boosters are practically tied at the top."),
        ("Searched hyperparameter grids with GridSearchCV on the training folds only, and compared before vs after "
         "on cross-validation and on the untouched test set.", tuning_result),
        ("Selected the model on cross-validation, chose a decision threshold from out-of-fold predictions, and "
         "exposed everything in an interactive scoring tool. Tuned pipelines are saved for reuse.",
         f"{best} at threshold {fmt3(tuned.threshold)} catches {tuned.metrics['Recall']:.0%} of test churners. "
         "Ready for exploration and teaching; not validated for production banking decisions."),
    ]
    did, found = details[current]
    with st.container(border=True):
        c1, c2, c3 = st.columns([5, 5, 2], gap="medium", vertical_alignment="center")
        c1.markdown(f"**{JOURNEY[current][1]} Step {current + 1} · {stage}: what we did**\n\n{did}")
        c2.markdown(f"**What we measured**\n\n{found}")
        target = PAGES[JOURNEY[current][2]]
        c3.button(f"Open Section {target.split('.')[0]} →", key=f"journey_go_{current}", on_click=_go_to,
                  args=(target,), type="primary", width="stretch", help=f"Go to: {target}")

    st.info("**How to read this dashboard.** Use the sidebar to move between sections and to change the class "
            "weighting and decision threshold; every chart updates live. Hover any metric or column header for a "
            "definition, and open **Key terms** panels or the sidebar **Glossary** for more.", icon="🧭")

# =============================================================================
# 1. Project overview & data inspection
# =============================================================================
def page_overview() -> None:
    df = A.df
    page_header(1, "The business problem, the data behind it, and the checks that make the data safe to model.")

    st.markdown(
        "#### Business objective\n"
        "Acquiring a new bank customer typically costs more than retaining an existing one. The goal of this "
        "project is to **predict which customers are likely to leave the bank (`Exited = 1`)**, so that a "
        "retention team could prioritise who to contact. Five models are compared on identical data: an "
        "interpretable linear baseline (**Logistic Regression**), a bagging ensemble (**Random Forest**) and three "
        "gradient-boosting implementations (**XGBoost**, **LightGBM**, **CatBoost**). Random Forest and XGBoost "
        "are also tuned with GridSearchCV (Section 4)."
    )

    churn_rate = df[config.TARGET].mean()
    k = st.columns(4)
    k[0].metric("Customers", f"{len(df):,}", help="Rows in churn_modelling.csv, one per customer", border=True)
    k[1].metric("Columns", f"{df.shape[1]}", help="10 model features, 2 identifiers, 1 target", border=True)
    k[2].metric("Churn rate", f"{churn_rate:.2%}", help="Share of customers with Exited = 1. " + define("Class imbalance"),
                border=True)
    k[3].metric("Model features", f"{len(config.FEATURES)}",
                help=f"{len(config.NUMERIC_FEATURES)} numeric, {len(config.CATEGORICAL_FEATURES)} categorical", border=True)

    what_why_how(
        "Loaded the CSV with pandas, profiled every column, and ran automated health checks (missing values, "
        "duplicates, identifier uniqueness, target validity) before any modelling.",
        "Models silently learn from whatever they are given. A duplicated row, a leaked identifier, or a target "
        "with unexpected values would make every downstream metric untrustworthy.",
        f"All checks pass. The one material property is <b>class imbalance</b>: only {churn_rate:.1%} of customers "
        "churned, so accuracy alone is misleading and evaluation focuses on the churn class.",
    )

    key_terms("Class imbalance", "Stratification", "Train/test split", "Data leakage")
    t1, t2, t3, t4 = st.tabs(["Feature dictionary", "Data health checks", "Class balance", "Churn by segment"])

    with t1:
        rows = []
        for col in df.columns:
            role, desc = FEATURE_INFO[col]
            s = df[col]
            if pd.api.types.is_numeric_dtype(s) and col != "CustomerId":
                values = f"{s.min():,.2f} – {s.max():,.2f}" if s.dtype.kind == "f" else f"{s.min():,} – {s.max():,}"
            elif col in config.CATEGORICAL_FEATURES:
                values = ", ".join(sorted(s.unique()))
            else:
                values = f"{s.nunique():,} unique values"
            rows.append({"Column": col, "Role": role, "dtype": str(s.dtype), "Unique": s.nunique(),
                         "Values / range": values, "Description": desc})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown(
            "**Why these roles?** Identifiers are dropped because a customer number or surname cannot generalise "
            "to new customers; a model could only memorise them. `HasCrCard` and `IsActiveMember` are already "
            "0/1 flags, so they can stay numeric. `Geography` and `Gender` are text categories with no natural "
            "order, so they are one-hot encoded (Section 2)."
        )

    with t2:
        target_values = sorted(df[config.TARGET].unique().tolist())
        zero_balance = (df["Balance"] == 0).mean()
        checks = [
            ("Shape matches expectation", f"{df.shape[0]:,} rows × {df.shape[1]} columns (expected 10,000 × 13)",
             df.shape == (10000, 13)),
            ("Required columns present", f"{len(config.REQUIRED_COLUMNS)} of {len(config.REQUIRED_COLUMNS)} found",
             all(c in df.columns for c in config.REQUIRED_COLUMNS)),
            ("Missing values", f"{int(df.isna().sum().sum())} missing cells", not df.isna().any().any()),
            ("Duplicate rows", f"{int(df.duplicated().sum())} duplicates", not df.duplicated().any()),
            ("CustomerId unique", f"{df['CustomerId'].nunique():,} unique of {len(df):,}", df["CustomerId"].is_unique),
            ("Target is binary", f"Values found: {target_values}", target_values == [0, 1]),
            ("Expected category levels", "Geography: 3 levels · Gender: 2 levels",
             df["Geography"].nunique() == 3 and df["Gender"].nunique() == 2),
        ]
        health = pd.DataFrame([{"Check": c, "Result": r, "Status": "✅ Pass" if ok else "❌ Fail"} for c, r, ok in checks])
        notes = pd.DataFrame([
            {"Check": "Class balance", "Result": f"{churn_rate:.2%} churners", "Status": "⚠️ Imbalanced: handled in evaluation"},
            {"Check": "Zero balances", "Result": f"{zero_balance:.1%} of customers have Balance = 0",
             "Status": "ℹ️ Plausible real value, kept as-is"},
        ])
        st.dataframe(pd.concat([health, notes]), hide_index=True, width="stretch")
        st.markdown(
            "These checks are also enforced in code: `churn/data.py::validate_dataset` stops the pipeline with a "
            "clear error if columns are missing, the target is not strictly 0/1, or features contain missing "
            "values, rather than producing misleading results. No imputation is applied because the data does "
            "not need it."
        )

    with t3:
        counts = df[config.TARGET].value_counts().sort_index()
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure(go.Bar(
                x=["Stayed (0)", "Exited (1)"], y=counts.values,
                marker_color=[CLASS_COLORS["Stayed"], CLASS_COLORS["Exited"]],
                text=[f"{v:,}<br>({v / counts.sum():.1%})" for v in counts.values], textposition="outside",
                hovertemplate="%{x}: %{y:,} customers<extra></extra>",
            ))
            fig.update_layout(yaxis=dict(title="Customers", range=[0, counts.max() * 1.18]), showlegend=False)
            plotly(fig, 340)
        with c2:
            st.markdown(
                f"**What it shows.** {counts[0]:,} customers stayed and {counts[1]:,} left: roughly a 4 : 1 ratio.\n\n"
                f"**Why it matters.** A model that predicts *everyone stays* would already be "
                f"**{counts[0] / counts.sum():.1%} accurate** while finding zero churners. That is why this "
                "dashboard emphasises Precision, Recall, F1 and ROC-AUC for the churn class, and why the "
                "train/test split and the cross-validation folds are **stratified**.\n\n"
                "**Handling it.** Two levers are available in the sidebar: **class weighting** (changes how the "
                "models are trained) and **threshold strategy** (changes how probabilities become decisions)."
            )

    with t4:
        st.markdown(
            "Observed churn rate per customer segment, computed on the **training set only**, so that "
            "exploration never peeks at the held-out test customers."
        )
        options = {"Geography": "Geography", "Gender": "Gender", "NumOfProducts": "Number of products",
                   "IsActiveMember": "Active member", "HasCrCard": "Has credit card", "Tenure": "Tenure (years)",
                   "Age": "Age band"}
        choice = st.selectbox("Segment by", list(options), format_func=options.get,
                              help="Churn rate within each group of customers. Segments with n < "
                                   f"{MIN_SEGMENT_N} are flagged because their rates are unstable.")
        stats = segment_stats(A.X_train, A.y_train, choice)
        fig = go.Figure(go.Bar(
            x=stats.index.astype(str), y=stats["rate"],
            marker_color=[BASELINE if small else CLASS_COLORS["Exited"] for small in stats["small"]],
            marker_pattern_shape=["/" if small else "" for small in stats["small"]],
            text=[f"{r:.1%}<br>n = {n:,}{' ⚠' if small else ''}" for r, n, small in zip(stats["rate"], stats["n"], stats["small"])],
            textposition="outside",
            customdata=np.column_stack([stats["n"], stats["churned"]]),
            hovertemplate="%{x}: %{y:.1%} churn<br>%{customdata[1]:,} of %{customdata[0]:,} customers churned<extra></extra>",
        ))
        fig.add_hline(y=A.base_rate, line_dash="dash", line_color=BASELINE,
                      annotation_text=f"Overall {A.base_rate:.1%}", annotation_position="top left")
        fig.update_layout(xaxis=dict(title=options[choice], type="category"),
                          yaxis=dict(title="Churn rate", tickformat=".0%", range=[0, max(stats["rate"].max() * 1.3, 0.3)]))
        plotly(fig, 380)
        st.dataframe(
            pd.DataFrame({"Customers (n)": stats["n"], "Churned": stats["churned"], "Churn rate": stats["rate"],
                          "Sample size": np.where(stats["small"], f"⚠️ small (n < {MIN_SEGMENT_N})", "ok")})
            .style.format({"Customers (n)": "{:,}", "Churned": "{:,}", "Churn rate": "{:.1%}"}),
            width="stretch",
        )
        small = stats[stats["small"]]
        if not small.empty:
            st.warning(
                f"Small-sample segments (n < {MIN_SEGMENT_N}): "
                + "; ".join(f"**{lvl}**: {int(r['churned'])} of {int(r['n'])} churned" for lvl, r in small.iterrows())
                + ". Rates from so few customers are unstable (shown hatched in gray); a 0% or 100% rate should not "
                "be read as a certainty.", icon="⚠️")
        st.info("These are **descriptive associations**, not causes. A segment with higher churn does not prove "
                "that the attribute makes customers leave.", icon="ℹ️")


# =============================================================================
# 2. Preprocessing & pipeline architecture
# =============================================================================
def page_pipeline() -> None:
    page_header(2, "How raw data becomes model input, step by step, and how the design rules out data leakage.")
    what_why_how(
        "Selected features by explicit name, split the data 80/20 with stratification, and wrapped one-hot "
        "encoding, scaling and each classifier in a single scikit-learn <code>Pipeline</code>.",
        "Every preprocessing step <i>learns</i> something (categories, means, standard deviations). If it learns "
        "from the test rows, the test score becomes optimistically biased; this is <b>data leakage</b>.",
        "Read the diagram left to right: the split happens first, everything that is fitted sits on the "
        "training branch, and the test branch only ever calls <code>transform</code> / <code>predict</code>.",
    )

    key_terms("Data leakage", "One-hot encoding", "Standardisation", "Stratification", "Cross-validation (CV)",
              "Out-of-fold (OOF) prediction")
    st.subheader("Pipeline architecture")
    st.graphviz_chart(
        f"""
digraph G {{
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=15, margin="0.18,0.1",
        fillcolor="#eef3fb", color="#2a78d6", fontcolor="#0b0b0b"];
  edge [color="#8a8984", penwidth=1.4];
  data [label="churn_modelling.csv\\n{len(A.df):,} rows\\nvalidate · select X\\n(drop IDs, target)"];
  split [label="Stratified split\\n80 / 20 · seed {config.RANDOM_STATE}", fillcolor="#fdf0e9", color="#eb6834"];
  train [label="Train\\n{len(A.X_train):,} rows"];
  test [label="Test\\n{len(A.X_test):,} rows", fillcolor="#f2f2f0", color="#8a8984"];
  cv [label="{config.CV_FOLDS}-fold stratified CV\\nscores · OOF probs\\n→ threshold", fillcolor="#fdf0e9", color="#eb6834"];
  subgraph cluster_p {{ label="Pipeline.fit(X_train, y_train)"; fontname="Helvetica"; fontsize=14; fontcolor="#8a8984";
    color="#2a78d6"; style="dashed,rounded";
    ct [label="ColumnTransformer\\nOneHot + Scaler*"]; clf [label="Classifier\\nLogReg / RF / XGBoost\\nLightGBM / CatBoost"]; ct -> clf; }}
  pred [label="predict_proba(X_test)\\ntransform only", fillcolor="#f2f2f0", color="#8a8984"];
  ev [label="Metrics\\nCM · ROC · importance", fillcolor="#f2f2f0", color="#8a8984"];
  data -> split; split -> train; split -> test; train -> cv; train -> ct; clf -> pred; cv -> pred; test -> pred -> ev;
}}""",
        width="stretch",
    )
    st.caption("*StandardScaler is applied in the Logistic Regression pipeline only. Orange = split/CV steps; gray = test-only path.")

    st.subheader("Step 1: Explicit feature selection")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Numeric features (8)**\n\n" + "\n".join(f"- `{c}`" for c in config.NUMERIC_FEATURES))
    with c2:
        st.markdown("**Categorical features (2)**\n\n" + "\n".join(f"- `{c}`" for c in config.CATEGORICAL_FEATURES))
        st.markdown("**Excluded:** `CustomerId`, `Surname` (identifiers) and `Exited` (the target).")
    st.markdown(
        "`X` is built from a named column list (`config.FEATURES`), never from *all columns but the last*. "
        "If the CSV column order ever changes, the code still selects the right inputs, and explicit checks "
        "raise an error if the target or an identifier ever appears in `X`."
    )

    st.subheader("Step 2: Stratified train/test split")
    # The same split call without stratify=y, for comparison only (not used anywhere else).
    _, y_test_plain = train_test_split(A.df[config.TARGET], test_size=config.TEST_SIZE,
                                       random_state=config.RANDOM_STATE)
    split_df = pd.DataFrame([
        {"Set": "Full data", "Rows": len(A.df), "Churners": int(A.df[config.TARGET].sum()), "Churn rate": A.df[config.TARGET].mean()},
        {"Set": "Train (stratified)", "Rows": len(A.y_train), "Churners": int(A.y_train.sum()), "Churn rate": A.y_train.mean()},
        {"Set": "Test (stratified)", "Rows": len(A.y_test), "Churners": int(A.y_test.sum()), "Churn rate": A.y_test.mean()},
        {"Set": "Test, if NOT stratified (for comparison)", "Rows": len(y_test_plain), "Churners": int(y_test_plain.sum()),
         "Churn rate": y_test_plain.mean()},
    ])
    st.dataframe(split_df.style.format({"Churn rate": "{:.2%}", "Rows": "{:,}", "Churners": "{:,}"}), hide_index=True, width="stretch")
    st.markdown(
        "`train_test_split(..., test_size=0.2, stratify=y, random_state=42)` samples each class separately, so "
        "both sets keep the ~20.4% churn rate. Without stratification the test churn rate drifts by chance (last "
        "row), which makes metrics noisier and less comparable. The fixed `random_state` makes the split "
        "identical on every run, and **all models use this same split**."
    )

    lr_pre = A.models["Logistic Regression"].named_steps["preprocess"]
    st.subheader("Step 3: One-hot encoding of categorical features")
    sample = A.X_train[config.CATEGORICAL_FEATURES].head(4)
    encoder = lr_pre.named_transformers_["cat"].named_steps["onehot"]
    encoded = pd.DataFrame(encoder.transform(sample).toarray().astype(int),
                           columns=encoder.get_feature_names_out(config.CATEGORICAL_FEATURES), index=sample.index)
    c1, c2 = st.columns([2, 3])
    c1.markdown("**Input (training rows)**")
    c1.dataframe(sample, width="stretch")
    c2.markdown("**Output of the fitted `OneHotEncoder`**")
    c2.dataframe(encoded, width="stretch")
    st.markdown(
        "Each category becomes its own 0/1 column. Encoding `France = 0, Germany = 1, Spain = 2` instead would "
        "imply a false order and distance that a linear model would take literally. "
        "`handle_unknown=\"ignore\"` means a category never seen in training is encoded as all zeros instead of "
        "crashing at prediction time. The `ColumnTransformer` always returns a **dense** array "
        "(`sparse_threshold=0`): XGBoost treats entries absent from a sparse matrix as *missing*, which would "
        "silently corrupt one-hot zeros and `Balance = 0`."
    )

    st.subheader("Step 4: Standardisation (Logistic Regression only)")
    st.latex(r"z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}")
    scaler = lr_pre.named_transformers_["num"].named_steps["scaler"]
    scale_df = pd.DataFrame({"Feature": config.NUMERIC_FEATURES, "Learned mean μ": scaler.mean_,
                             "Learned std σ": scaler.scale_})
    c1, c2 = st.columns([3, 2])
    c1.dataframe(scale_df.style.format({"Learned mean μ": "{:,.3f}", "Learned std σ": "{:,.3f}"}), hide_index=True, width="stretch")
    c2.markdown(
        "**Why scale for Logistic Regression?** Raw features differ by orders of magnitude (`Balance` up to "
        "~250,000 vs `NumOfProducts` 1–4). The L2 penalty and the gradient-based solver treat all coefficients "
        "on one scale, so unscaled inputs distort both regularisation and convergence.\n\n"
        "**Why not for the tree models** (Random Forest, XGBoost, LightGBM, CatBoost)? A tree split asks *is x ≤ c?*. Any monotonic rescaling "
        "moves *c* with the data and leaves the splits unchanged, so scaling has no effect."
    )

    st.subheader("Step 5: How data leakage is prevented")
    feats = ["Age", "Balance", "CreditScore"]
    check = pd.DataFrame({
        "Feature": feats,
        "Scaler's learned mean": [scaler.mean_[config.NUMERIC_FEATURES.index(c)] for c in feats],
        "Training-set mean": [A.X_train[c].mean() for c in feats],
        "Full-data mean": [A.df[c].mean() for c in feats],
        "Test-set mean": [A.X_test[c].mean() for c in feats],
    })
    st.dataframe(check.style.format({c: "{:,.4f}" for c in check.columns[1:]}), hide_index=True, width="stretch")
    st.markdown(
        "**Live proof:** the scaler's learned means equal the *training* means exactly, and differ from the "
        "full-data and test means. The safeguards in the code are:\n"
        "1. **Split first.** `make_train_test_split` runs before anything is fitted.\n"
        "2. **Preprocessing inside the Pipeline.** `Pipeline.fit(X_train, y_train)` fits the encoder, scaler "
        "and classifier on training rows only; `predict` on test rows only *applies* them.\n"
        "3. **Cross-validation refits the whole pipeline in every fold**, so even the held-out fold never "
        "influences the scaler or encoder it is scored with.\n"
        "4. **No test-set tuning.** Optimised thresholds and class weights come from training data only.\n"
        "5. **Automated tests** (`tests/`) assert these properties on every run."
    )
    with st.expander("Inspect the fitted XGBoost pipeline object"):
        st.code(repr(A.models["XGBoost"]), language="python")


# =============================================================================
# 3. Model explorer & comparison
# =============================================================================
def page_models() -> None:
    page_header(3, f"{len(A.models)} models, the same {len(A.y_test):,} held-out test customers, and the configuration selected in the sidebar.")
    st.markdown(CONFIG_LINE)
    what_why_how(
        "Scored each fitted pipeline on the untouched test set and converted probabilities to decisions with "
        "the selected thresholds. Positive class = churn (<code>Exited = 1</code>).",
        "Precision, Recall and F1 describe the churn class that matters to the business; ROC-AUC measures "
        "ranking quality independent of any threshold. Accuracy is shown but is secondary.",
        "Bold = best in column. Changing the threshold trades Precision for Recall but leaves ROC-AUC unchanged. "
        "The analytical summary at the bottom explains <b>which model is best and why</b>.",
    )

    key_terms("Precision", "Recall", "F1", "Accuracy", "ROC_AUC", "Threshold", "Confusion matrix", "ROC curve",
              "Precision–Recall curve", "Average Precision", "Operating point")
    t1, t2, t3, t4, t5 = st.tabs(["Metrics", "Confusion matrices", "ROC curves", "Precision–Recall", "Strategy comparison"])

    with t1:
        st.caption("Hover a column header for its definition.")
        metrics_dataframe(TABLE)
        fig = go.Figure()
        for name, row in TABLE.iterrows():
            fig.add_bar(name=name, x=METRIC_COLS, y=row.values, marker_color=MODEL_COLORS[name],
                        text=[fmt3(v) for v in row.values], textposition="outside", customdata=hover3(row.values),
                        hovertemplate="%{x}: %{customdata[0]}<extra>%{fullData.name}</extra>")
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06, yaxis=dict(range=[0, 1.08], title="Score"),
                          legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
        plotly(fig, 400)
        st.caption(f"Reference: predicting “stayed” for every test customer gives Accuracy {(A.y_test == 0).mean():.2%} and Recall 0.")

    with t2:
        items = list(EVALS.items())
        cols = [c for start in range(0, len(items), 3) for c in st.columns(3, gap="medium")]
        for col, (name, e) in zip(cols, items):
            tn, fp, fn, tp = e.confusion.ravel()
            with col:
                plotly(confusion_figure(e.confusion, f"{name} · t = {fmt3(e.threshold)}"), 360)
                st.markdown(
                    f"Caught **{tp}** of {tp + fn} churners (missed **{fn}**). Flagged **{fp}** of {tn + fp} "
                    f"loyal customers. Of {tp + fp} flagged, {tp / max(tp + fp, 1):.1%} truly churned."
                )
        st.markdown(
            "**Reading the matrix.** Rows are the truth, columns the prediction. **False negatives** (bottom-left) "
            "are churners the bank would not contact; **false positives** (top-right) are retention offers spent "
            "on customers who would have stayed. Which error is more expensive is a business decision."
        )

    with t3:
        fig = go.Figure()
        for name, scores in A.test_scores.items():
            fpr, tpr, _ = roc_curve(A.y_test, scores, pos_label=1)
            e = EVALS[name]
            tn, fp, fn, tp = e.confusion.ravel()
            fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=MODEL_COLORS[name], width=2),
                            name=f"{name} (AUC = {fmt3(e.metrics['ROC_AUC'])})", customdata=hover3(fpr, tpr),
                            hovertemplate="FPR %{customdata[0]} · TPR %{customdata[1]}<extra></extra>")
            op_x, op_y = [fp / (fp + tn)], [tp / (tp + fn)]
            fig.add_scatter(x=op_x, y=op_y, mode="markers",
                            marker=dict(size=12, color=MODEL_COLORS[name], line=dict(width=2, color="white")),
                            name=f"{name} operating point (t = {fmt3(e.threshold)})", customdata=hover3(op_x, op_y),
                            hovertemplate="FPR %{customdata[0]} · TPR %{customdata[1]}<extra></extra>")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=BASELINE, dash="dash", width=1.5),
                        name="Random classifier (AUC = 0.500)", hoverinfo="skip")
        fig.update_layout(xaxis=dict(title="False positive rate (loyal customers flagged)", range=[0, 1]),
                          yaxis=dict(title="True positive rate / Recall (churners caught)", range=[0, 1.01]),
                          legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom"))
        plotly(fig, 480)
        st.markdown(
            "The ROC curve plots every possible threshold at once. **AUC** is the probability that a randomly "
            "chosen churner receives a higher score than a randomly chosen non-churner (0.5 = coin flip, "
            "1.0 = perfect ranking). The dots show where the selected thresholds sit on each curve."
        )

    with t4:
        fig = go.Figure()
        prevalence = A.y_test.mean()
        for name, scores in A.test_scores.items():
            precision, recall, _ = precision_recall_curve(A.y_test, scores, pos_label=1)
            ap = average_precision_score(A.y_test, scores, pos_label=1)
            e = EVALS[name]
            fig.add_scatter(x=recall, y=precision, mode="lines", line=dict(color=MODEL_COLORS[name], width=2),
                            name=f"{name} (AP = {fmt3(ap)})", customdata=hover3(recall, precision),
                            hovertemplate="Recall %{customdata[0]} · Precision %{customdata[1]}<extra></extra>")
            fig.add_scatter(x=[e.metrics["Recall"]], y=[e.metrics["Precision"]], mode="markers",
                            marker=dict(size=12, color=MODEL_COLORS[name], line=dict(width=2, color="white")),
                            name=f"{name} operating point", hoverinfo="skip")
        fig.add_hline(y=prevalence, line_dash="dash", line_color=BASELINE,
                      annotation_text=f"No-skill baseline = churn rate {prevalence:.1%}", annotation_position="bottom left")
        fig.update_layout(xaxis=dict(title="Recall", range=[0, 1]), yaxis=dict(title="Precision", range=[0, 1.02]),
                          legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"))
        plotly(fig, 480)
        st.markdown(
            "With imbalanced classes the Precision–Recall curve is often more informative than ROC, because "
            "its baseline is the churn rate rather than 0.5. **Average Precision (AP)** summarises the curve; "
            "it is shown as supplementary information and is not used for any selection."
        )

    with t5:
        rows = []
        for s in STRATEGIES:
            for name, e in A.evaluate_strategy(s, custom_t).items():
                tn, fp, fn, tp = e.confusion.ravel()
                rows.append({"Strategy": s if s != STRATEGY_CUSTOM else f"Custom (t = {custom_t:.2f})", "Model": name,
                             "Threshold": e.threshold, **{m: e.metrics[m] for m in METRIC_COLS},
                             "Churners caught (TP)": tp, "False alarms (FP)": fp, "Missed (FN)": fn})
        st.dataframe(pd.DataFrame(rows).style.format({c: fmt3 for c in ["Threshold"] + METRIC_COLS}),
                     hide_index=True, width="stretch", column_config=column_help(["Threshold"] + METRIC_COLS))
        default, tuned = A.evaluate_strategy(STRATEGY_DEFAULT), A.evaluate_strategy(STRATEGY_OPTIMIZED)
        bullets([
            f"**{n}:** threshold 0.500 → {fmt3(tuned[n].threshold)}. Recall {fmt3(default[n].metrics['Recall'])} → "
            f"{fmt3(tuned[n].metrics['Recall'])} ({int(tuned[n].confusion[1, 1] - default[n].confusion[1, 1]):+d} churners caught), "
            f"Precision {fmt3(default[n].metrics['Precision'])} → {fmt3(tuned[n].metrics['Precision'])} "
            f"({int(tuned[n].confusion[0, 1] - default[n].confusion[0, 1]):+d} false alarms), "
            f"F1 {fmt3(default[n].metrics['F1'])} → {fmt3(tuned[n].metrics['F1'])}."
            for n in A.models
        ])
        st.markdown(
            "CV-optimized thresholds were chosen **before** looking at these test results, from out-of-fold "
            "predictions on the training set, so their test scores are honest. Custom-threshold rows are exploratory."
        )

    st.subheader("Automated analytical summary: which model performs best, and why")
    cv = A.cv_summary()
    summary = analytical_summary(TABLE, cv, A.cv_folds, A.X_train, A.y_train, get_permutation(class_weight))
    best = best_model(cv)  # selected on cross-validation (training set only)
    confirmed = TABLE["ROC_AUC"].idxmax() == best
    confirmation = ("the highest of all models, so the test set confirms the choice" if confirmed
                    else "<b>not</b> the highest, so the test set does not confirm the choice")
    st.markdown(
        f'<div class="verdict best"><h3>Selected model: {best}</h3><p><b>Selection:</b> highest mean ROC-AUC in '
        f'{config.CV_FOLDS}-fold cross-validation on the training set ({fmt3(cv.loc[best, "ROC_AUC mean"])} ± '
        f'{fmt3(cv.loc[best, "ROC_AUC std"])}). <b>Final confirmation</b> on the untouched test set: ROC-AUC '
        f'{fmt3(TABLE.loc[best, "ROC_AUC"])}, {confirmation}. The test set was not used to make the selection.</p></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Verdict (measured)**")
        bullets(summary["verdict"])
        st.markdown("**Robustness across CV folds (measured)**")
        bullets(summary["evidence"])
    with c2:
        st.markdown("**Why (measured patterns → algorithm explanation)**")
        bullets(summary["why"])
    with st.expander("Caveats"):
        bullets(summary["caveats"])


# =============================================================================
# 4. Hyperparameter tuning: before vs after
# =============================================================================
def _grid_display(grid: pd.DataFrame) -> pd.DataFrame:
    """Grid CSV values as text, with empty cells shown as None (e.g. max_depth=None)."""
    params = [c for c in grid.columns if c not in ("CV ROC-AUC mean", "CV ROC-AUC std", "Rank")]
    out = grid.copy()
    for c in params:
        out[c] = out[c].astype(str).replace({"": "None", "nan": "None"})
        out[c] = out[c].str.replace(r"\.0$", "", regex=True)
    return out[params + ["CV ROC-AUC mean", "CV ROC-AUC std", "Rank"]]


def page_tuning() -> None:
    page_header(4, "GridSearchCV on the training folds: what changed, did it help, and which model is best overall?")
    T = get_tuning(tuning_stamp())
    if T is None:
        st.info("No tuning results found in `outputs/`. Run `python main.py` once (the grid search takes several "
                "minutes); this page then loads the saved tuned models.", icon="ℹ️")
        return
    B = A if class_weight is None else get_analysis(None)  # tuning used equal class weights
    if class_weight is not None:
        st.info("Tuning was run with equal class weights, so this page always compares against the unweighted "
                "baselines, whatever the sidebar weighting.", icon="ℹ️")

    what_why_how(
        f"Searched {', '.join(f'{n_candidates(PARAM_GRIDS[m])} {m}' for m in TUNED_MODELS)} hyperparameter "
        f"combinations with <code>GridSearchCV</code>, scored by ROC-AUC in the same {config.CV_FOLDS} training "
        "folds as the baselines. The best candidate was refitted on all training data and tested once.",
        "Baseline settings are sensible defaults, not optimal for this data. Tuning shows how much performance "
        "was left on the table, and whether a model's rank changes once each is given a fair search.",
        "Compare hatched (before) with solid (after) bars. CV gains are slightly optimistic because the winner was "
        "picked on those folds; the <b>test</b> columns are the fair before-vs-after comparison.",
    )
    key_terms("Hyperparameter", "GridSearchCV", "Baseline", "Cross-validation (CV)", "Selection bias (tuning)",
              "Nested cross-validation", "ROC_AUC", "Precision", "Recall", "Paired comparison")

    base_cv = B.cv_summary()
    tuned_cv = summarize_cv(T["tuned_folds"])
    base_test = Analysis.metrics_table(B.evaluate_strategy(STRATEGY_DEFAULT))
    tuned_pred = predict_models(T["models"], B.X_test)
    tuned_test = build_comparison_table(B.y_test, tuned_pred)
    comparison = comparison_table(base_cv, tuned_cv, base_test, tuned_test)
    cv_c, test_c = candidates_table(base_cv, tuned_cv), candidates_table(base_test, tuned_test)
    insights = tuning_insights(comparison, B.cv_folds, T["tuned_folds"], T["params"], cv_c, test_c)

    k = st.columns(2 * len(TUNED_MODELS))
    for i, name in enumerate(TUNED_MODELS):
        row = comparison[(comparison["Model"] == name) & (comparison["Metric"] == "ROC_AUC")].set_index("Stage")
        k[2 * i].metric(f"{name} · CV ROC-AUC", fmt3(row.loc["CV mean", "Tuned"]),
                        delta=f"{fmt3_signed(row.loc['CV mean', 'Change'])} vs baseline", border=True,
                        help=f"Mean over {config.CV_FOLDS} training folds. Baseline {fmt3(row.loc['CV mean', 'Baseline'])}. "
                             + define("Selection bias (tuning)"))
        k[2 * i + 1].metric(f"{name} · Test ROC-AUC", fmt3(row.loc["Test", "Tuned"]),
                            delta=f"{fmt3_signed(row.loc['Test', 'Change'])} vs baseline", border=True,
                            help=f"Untouched test set. Baseline {fmt3(row.loc['Test', 'Baseline'])}. " + define("ROC_AUC"))

    best = cv_c["ROC_AUC mean"].idxmax()
    st.markdown(f'<div class="verdict best"><h3>Best model overall: {best}</h3><p>'
                + " ".join(line.replace("**", "") for line in insights["best"]) + "</p></div>", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["Before vs after", "Hyperparameters", "Search results", "All models compared",
                                  "Best model & trade-offs"])

    with t1:
        stage = st.segmented_control("Evaluated on", ["CV mean", "Test"], default="Test", key="tune_stage",
                                     help="CV mean: average over the training folds (slightly optimistic for the "
                                          "tuned models). Test: the untouched test set at threshold 0.5.") or "Test"
        data = comparison[comparison["Stage"] == stage]
        metrics = ["Precision", "Recall", "F1", "Accuracy", "ROC_AUC"]
        fig = go.Figure()
        for name in TUNED_MODELS:
            d = data[data["Model"] == name].set_index("Metric").loc[metrics]
            for version, opacity, pattern in [("Baseline", 0.5, "/"), ("Tuned", 1.0, "")]:
                fig.add_bar(name=f"{name} · {version.lower()}", x=[m.replace("_", "-") for m in metrics], y=d[version],
                            marker=dict(color=MODEL_COLORS[name], opacity=opacity, pattern_shape=pattern),
                            text=[fmt3(v) for v in d[version]], textposition="outside",
                            customdata=np.column_stack([hover3(d[version])[:, 0], [fmt3_signed(c) for c in d["Change"]]]),
                            hovertemplate="%{x}: %{customdata[0]} (change %{customdata[1]})<extra>%{fullData.name}</extra>")
        fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.05, yaxis=dict(range=[0, 1.1], title="Score"),
                          legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
        plotly(fig, 420)
        table = data.assign(Metric=data["Metric"].str.replace("_", "-")).drop(columns="Stage")
        st.dataframe(table.style.format({"Baseline": fmt3, "Tuned": fmt3, "Change": fmt3_signed}), hide_index=True,
                     width="stretch", column_config={
                         "Baseline": st.column_config.Column(help=define("Baseline")),
                         "Tuned": st.column_config.Column(help="The GridSearchCV winner, refitted on all training data."),
                         "Change": st.column_config.Column(help="Tuned − Baseline. Positive = better.")})

        st.markdown("**Fold by fold (ROC-AUC, the tuning metric):** the same folds before and after, so each pair is "
                    "directly comparable.")
        fig = go.Figure()
        for name in TUNED_MODELS:
            folds = list(T["tuned_folds"][name].index)
            for version, values, dash in [("baseline", B.cv_folds[name]["ROC_AUC"], "dash"),
                                          ("tuned", T["tuned_folds"][name]["ROC_AUC"], "solid")]:
                fig.add_scatter(x=folds, y=values, mode="lines+markers", name=f"{name} · {version}",
                                line=dict(color=MODEL_COLORS[name], dash=dash, width=2), marker=dict(size=8),
                                customdata=hover3(values), hovertemplate="%{x}: %{customdata[0]}<extra>%{fullData.name}</extra>")
        fig.update_layout(yaxis=dict(title="CV ROC-AUC"), legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
        plotly(fig, 340)

    with t2:
        params = T["params"].copy()
        params["Meaning"] = params["Hyperparameter"].map(HYPERPARAMETERS)
        params["Changed"] = params["Changed"].map({True: "✅ changed", False: "kept baseline"})
        st.dataframe(params, hide_index=True, width="stretch", column_config={
            "Hyperparameter": st.column_config.Column(help=define("Hyperparameter")),
            "Values searched": st.column_config.Column(help="Every combination of these values was tried (a grid)."),
            "Baseline": st.column_config.Column(help=define("Baseline")),
            "Tuned": st.column_config.Column(help="Value in the candidate with the highest mean CV ROC-AUC."),
            "Meaning": st.column_config.Column(width="large"),
        })
        st.markdown(
            "**Design of the grids.** Each grid contains the baseline values, so the search can only keep or improve "
            "on the baseline's CV score. The grids are deliberately small (training time on a laptop); the "
            "parameters chosen are the ones that most control model complexity: number of trees, depth, leaf size "
            "and learning rate."
        )
        with st.expander("See how n_estimators, learning_rate and depth work (animation)"):
            st.image(str(boosting_gif()), width="stretch")
            st.caption("Each frame adds one depth-2 tree, shrunk by the learning rate. Unseen-data loss stops "
                       "improving after ~15 trees: more trees, deeper trees or a larger learning rate fit noise sooner.")

    with t3:
        model = st.selectbox("Model", TUNED_MODELS, key="tune_model",
                             help="Every candidate in this model's grid, ranked by mean CV ROC-AUC.")
        grid = _grid_display(T["grid"].loc[model])
        baseline_vals = T["params"][T["params"]["Model"] == model].set_index("Hyperparameter")["Baseline"]
        is_base = np.all([grid[p].astype(str) == str(v) for p, v in baseline_vals.items()], axis=0)
        labels = [f"#{r} · " + ", ".join(f"{p}={row[p]}" for p in baseline_vals.index) for r, (_, row) in zip(grid["Rank"], grid.iterrows())]
        colors = [MODEL_COLORS[model] if r == 1 else (BASELINE if b else "rgba(138,137,132,0.35)")
                  for r, b in zip(grid["Rank"], is_base)]
        fig = go.Figure(go.Bar(
            y=labels[::-1], x=grid["CV ROC-AUC mean"][::-1], orientation="h", marker_color=colors[::-1],
            marker_pattern_shape=["/" if b else "" for b in is_base[::-1]],
            error_x=dict(type="data", array=grid["CV ROC-AUC std"][::-1], color=BASELINE, thickness=1),
            customdata=hover3(grid["CV ROC-AUC mean"][::-1], grid["CV ROC-AUC std"][::-1]),
            hovertemplate="%{y}<br>CV ROC-AUC %{customdata[0]} ± %{customdata[1]}<extra></extra>",
        ))
        lo = (grid["CV ROC-AUC mean"] - grid["CV ROC-AUC std"]).min()
        fig.update_layout(xaxis=dict(title="Mean CV ROC-AUC (± std over folds)", range=[lo - 0.005, None]),
                          yaxis=dict(automargin=True))
        plotly(fig, 140 + 24 * len(grid))
        st.caption("Coloured = selected (rank 1). Hatched gray = the baseline settings. Error bars = fold-to-fold std.")
        spread = grid["CV ROC-AUC mean"].max() - grid["CV ROC-AUC mean"].min()
        st.markdown(
            f"All {len(grid)} candidates lie within **{fmt3(spread)}** ROC-AUC of each other, while the fold-to-fold "
            f"standard deviation is about **{fmt3(grid['CV ROC-AUC std'].median())}**. Differences between nearby "
            "candidates are therefore mostly noise; the search mainly confirms which *region* of settings works."
        )

    with t4:
        metric = st.segmented_control("Metric", CV_METRICS, default="ROC_AUC", key="tune_metric",
                                      help="Compare all baseline and tuned models on one metric.") or "ROC_AUC"
        st.caption(f"**{metric.replace('_', '-')}:** {define(metric)}")
        names = list(cv_c.index)
        base_of = {n: n.replace(" (tuned)", "") for n in names}
        fig = go.Figure()
        fig.add_bar(x=names, y=cv_c[f"{metric} mean"], name="CV mean ± std",
                    marker=dict(color=[MODEL_COLORS[base_of[n]] for n in names],
                                pattern_shape=["" if n.endswith("(tuned)") else "/" for n in names]),
                    error_y=dict(type="data", array=cv_c[f"{metric} std"], color=BASELINE, thickness=1),
                    customdata=hover3(cv_c[f"{metric} mean"], cv_c[f"{metric} std"]),
                    hovertemplate="%{x}<br>CV %{customdata[0]} ± %{customdata[1]}<extra></extra>")
        fig.add_scatter(x=names, y=test_c[metric], mode="markers", name="Test set",
                        marker=dict(symbol="diamond", size=12, color="white", line=dict(width=2, color="#0b0b0b")),
                        customdata=hover3(test_c[metric]), hovertemplate="%{x}<br>Test %{customdata[0]}<extra></extra>")
        lo = min(cv_c[f"{metric} mean"].min(), test_c[metric].min())
        fig.update_layout(yaxis=dict(title=metric.replace("_", "-"), range=[max(lo - 0.1, 0), None]),
                          legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
        plotly(fig, 400)
        st.caption("Hatched = baseline settings, solid = tuned. ◇ = untouched test set (threshold 0.5). "
                   "The y-axis does not start at 0, to make small differences visible.")
        merged = pd.DataFrame({f"CV {m.replace('_', '-')}": format_mean_std(cv_c)[m] for m in ["Precision", "Recall", "F1", "ROC_AUC"]})
        for m in ["Precision", "Recall", "F1", "ROC_AUC"]:
            merged[f"Test {m.replace('_', '-')}"] = test_c[m].map(fmt3)
        st.dataframe(merged, width="stretch", column_config={
            c: st.column_config.Column(help=METRIC_HELP[c.split(" ")[1].replace("-", "_")]) for c in merged.columns})

        st.markdown("**ROC curves on the test set: baseline (dashed) vs tuned (solid)**")
        fig = go.Figure()
        curves = {n: B.test_scores[n] for n in B.models} | {tuned_name(n): p["y_score"] for n, p in tuned_pred.items()}
        for n, scores in curves.items():
            fpr, tpr, _ = roc_curve(B.y_test, scores, pos_label=1)
            tuned = n.endswith("(tuned)")
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{n} (AUC = {fmt3(test_c.loc[n, 'ROC_AUC'])})",
                            line=dict(color=MODEL_COLORS[base_of[n]], width=2.5 if tuned else 1.5,
                                      dash="solid" if tuned or base_of[n] not in TUNED_MODELS else "dash"),
                            customdata=hover3(fpr, tpr), hovertemplate="FPR %{customdata[0]} · TPR %{customdata[1]}<extra></extra>")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=BASELINE, dash="dot", width=1),
                        name="Random classifier (AUC = 0.500)", hoverinfo="skip")
        fig.update_layout(xaxis=dict(title="False positive rate", range=[0, 1]),
                          yaxis=dict(title="True positive rate / Recall", range=[0, 1.01]),
                          legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom", font=dict(size=11)))
        plotly(fig, 500)

    with t5:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**What tuning changed (measured)**")
            bullets(insights["per_model"])
            st.markdown("**Best model overall (measured)**")
            bullets(insights["best"])
        with c2:
            st.markdown("**Trade-offs: Precision, Recall and ROC-AUC (measured)**")
            bullets(insights["tradeoffs"])
        fig = go.Figure()
        for n in test_c.index:
            fig.add_scatter(x=[test_c.loc[n, "Recall"]], y=[test_c.loc[n, "Precision"]], mode="markers+text",
                            name=n, text=[n], textposition="top center", textfont=dict(size=11),
                            marker=dict(size=10 + 60 * (test_c.loc[n, "ROC_AUC"] - 0.7), color=MODEL_COLORS[base_of[n]],
                                        symbol="circle" if n.endswith("(tuned)") else "circle-open",
                                        line=dict(width=2, color=MODEL_COLORS[base_of[n]])),
                            customdata=[[fmt3(test_c.loc[n, m]) for m in ["Recall", "Precision", "ROC_AUC"]]],
                            hovertemplate="%{text}<br>Recall %{customdata[0]} · Precision %{customdata[1]} · "
                                          "ROC-AUC %{customdata[2]}<extra></extra>")
        fig.update_layout(showlegend=False, xaxis=dict(title="Recall (churners caught)"),
                          yaxis=dict(title="Precision (flags that are real churners)"))
        plotly(fig, 420)
        st.caption("Test set at threshold 0.5. Bigger marker = higher ROC-AUC. Open = baseline, filled = tuned. "
                   "Up and to the right is better; a model can only move along its own trade-off curve by changing "
                   "the threshold (Sections 3 and 8).")
        with st.expander("Caveats"):
            bullets(insights["caveats"])

# =============================================================================
# 5. Cross-validation & robustness
# =============================================================================
def page_cv() -> None:
    page_header(5, f"{config.CV_FOLDS}-fold stratified cross-validation on the training set: how good, and how stable, is each model?")
    st.markdown(f"Configuration: weighting **{weighting_label.split(' (')[0]}** (CV scores use the default 0.5 threshold).")
    what_why_how(
        f"Split the {len(A.y_train):,} training customers into {config.CV_FOLDS} stratified folds. For each fold, a "
        "fresh pipeline (preprocessing + model) was fitted on the other folds and scored on the held-out one.",
        "A single test split gives one number with unknown noise. k-fold CV gives <b>k</b> estimates: the mean "
        "is more reliable, and the standard deviation shows how much performance depends on which customers "
        "happen to be in the sample.",
        "Compare means, but always next to their ± std. A gap between models that is small relative to the "
        "std is not convincing; a model that wins in <i>every</i> fold is.",
    )

    key_terms("Cross-validation (CV)", "Mean ± std", "Stratification", "Paired comparison", "Overfitting",
              "Generalisation gap")
    st.subheader("How the folds are built")
    k = config.CV_FOLDS
    grid = np.array([[1 if i == j else 0 for j in range(k)] for i in range(k)])
    fig = go.Figure(go.Heatmap(
        z=grid, x=[f"Fold {j + 1}" for j in range(k)], y=[f"Round {i + 1}" for i in range(k)],
        colorscale=[[0, "rgba(42,120,214,0.35)"], [1, "#eb6834"]], showscale=False, xgap=4, ygap=4,
        text=[["validate" if v else "train" for v in row] for row in grid], texttemplate="%{text}",
        textfont=dict(color="white", size=13), hoverinfo="skip",
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    c1, c2 = st.columns([3, 2])
    with c1:
        plotly(fig, 260)
    with c2:
        st.markdown(
            f"Each round trains on {k - 1} folds (blue, ~{len(A.y_train) * (k - 1) // k:,} customers) and validates "
            f"on 1 fold (orange, ~{len(A.y_train) // k:,}). Every training customer is validated exactly once. "
            "Folds are **stratified** (each keeps the ~20% churn rate) and **shuffled with a fixed seed**, so all "
            "models see identical folds and their scores can be compared fold by fold."
        )

    st.subheader("Cross-validation scores (mean ± std)")
    summary = A.cv_summary()
    st.dataframe(format_mean_std(summary), width="stretch", column_config=column_help(CV_METRICS))

    metric = st.segmented_control("Metric", CV_METRICS, default="ROC_AUC", key="cv_metric",
                                  help="Choose the metric for the charts and tables below.") or "ROC_AUC"
    st.caption(f"**{metric.replace('_', '-')}:** {define(metric)}")
    fig = go.Figure()
    for name, folds in A.cv_folds.items():
        fig.add_box(y=folds[metric], name=name, marker_color=MODEL_COLORS[name], boxpoints="all", jitter=0.3,
                    pointpos=0, boxmean=True, text=list(folds.index), customdata=hover3(folds[metric]),
                    hovertemplate="%{text}: %{customdata[0]}<extra>%{fullData.name}</extra>")
        # CV scores use the default threshold, so compare with the test score at the default threshold too.
        test_value = A.evaluate(name, 0.5, use_predict=True).metrics[metric]
        fig.add_scatter(x=[name], y=[test_value], customdata=hover3([test_value]),
                        mode="markers", marker=dict(symbol="diamond", size=12, color="white",
                                                    line=dict(width=2, color=MODEL_COLORS[name])),
                        name=f"{name} test score", hovertemplate="Test set: %{customdata[0]}<extra></extra>",
                        showlegend=False)
    fig.update_layout(yaxis=dict(title=metric), showlegend=False)
    plotly(fig, 380)
    st.caption("Box = spread of the 5 fold scores (line = median, dashed = mean). ◇ = score on the held-out test set (threshold 0.5).")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Fold-by-fold scores**")
        fold_table = pd.DataFrame({name: f[metric] for name, f in A.cv_folds.items()})
        st.dataframe(styled_metrics(fold_table, axis=1), width="stretch")
        st.caption("Bold = best in the fold, decided on unrounded scores (two cells can look equal after rounding).")
    with c2:
        st.markdown("**Paired comparison (same folds)**")
        names = list(A.models)
        lines = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                p = paired_fold_comparison(A.cv_folds, a, b, metric)
                winner, loser = (a, b) if p["mean"] >= 0 else (b, a)
                wins = p["wins"] if p["mean"] >= 0 else p["n"] - p["wins"]
                lines.append(f"**{winner}** beats {loser} in **{wins}/{p['n']}** folds "
                             f"(mean Δ {fmt3(abs(p['mean']))} ± {fmt3(p['std'])})")
        bullets(lines)

    st.subheader("Generalisation check: training data vs cross-validation vs test set")
    test_default = Analysis.metrics_table(A.evaluate_strategy(STRATEGY_DEFAULT))
    gap = generalisation_table(summary, test_default, A.train_table, metric)
    within_col = f"Test within {GAP_STD_RULE} CV std"
    shown = gap.assign(**{within_col: gap[within_col].map({True: "✅ yes", False: "⚠️ no"})})
    st.dataframe(
        shown.style.format({**{c: fmt3 for c in ["Train", "CV mean", "CV std", "Test"]},
                            **{c: fmt3_signed for c in ["Train − CV", "Test − CV"]}}),
        width="stretch",
        column_config={
            "Train": st.column_config.Column(help="Final model scored on its own training data."),
            "CV mean": st.column_config.Column(help=define("Mean ± std")),
            "Train − CV": st.column_config.Column(help=define("Generalisation gap")),
            "Test − CV": st.column_config.Column(help="How far the test score is from the CV estimate."),
            within_col: st.column_config.Column(help="Is the test score within the usual fold-to-fold variation?"),
        },
    )
    st.caption(f"Train = final model scored on its own {len(A.y_train):,} training customers (threshold 0.5).")
    bullets(generalisation_notes(gap, metric))


# =============================================================================
# 6. Feature importance
# =============================================================================
def page_importance() -> None:
    page_header(6, "Which customer features do the models rely on, and how much should we trust each importance measure?")
    perm = get_permutation(class_weight)
    what_why_how(
        "Extracted the <b>built-in</b> importances of the four tree models (Random Forest: impurity decrease; "
        "XGBoost and LightGBM: gain; CatBoost: prediction change) and computed <b>permutation importance</b> for "
        f"all {len(A.models)} models on the test set ({PERMUTATION_REPEATS} repeats).",
        "Importance shows what a model has learned to use, which supports trust, debugging and communication "
        "with the business. Comparing two methods guards against the known biases of each.",
        "Longer bar = the model relies more on the feature. Built-in importance sums to 100% per model. "
        "Permutation importance is the drop in test ROC-AUC when the feature is shuffled; ≈ 0 means unused. "
        "Neither measure proves that a feature <i>causes</i> churn.",
    )

    key_terms("Built-in importance", "Permutation importance", "One-hot encoding", "ROC_AUC")
    t1, t2, t3 = st.tabs(["Built-in (tree models)", "Permutation (all models)", "Comparison & insights"])

    with t1:
        level = st.segmented_control("Level", ["Original features", "One-hot columns"], default="Original features",
                                     key="imp_level",
                                     help="One-hot columns shows each Geography/Gender value separately; original "
                                          "features sums them.") or "Original features"
        data = A.builtin_importance if level == "Original features" else A.builtin_importance_onehot
        cols = [c for _ in range(0, len(TREE_MODELS), 2) for c in st.columns(2, gap="large")]
        for col, name in zip(cols, TREE_MODELS):
            s = data[name].sort_values()
            fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=MODEL_COLORS[name],
                                   text=[f"{v:.1%}" for v in s.values], textposition="outside",
                                   hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
            fig.update_layout(title=dict(text=f"{name} · {IMPORTANCE_METHOD[name]}", x=0),
                              xaxis=dict(tickformat=".0%", range=[0, s.max() * 1.25], title="Share of total importance"))
            with col:
                plotly(fig, 120 + 30 * len(s))
        st.markdown(
            "**Random Forest (mean decrease in impurity):** every split reduces the Gini impurity of its node; "
            "a feature's importance is the total reduction it achieves, averaged over all trees.\n\n"
            "**XGBoost / LightGBM (gain):** the improvement in the training loss from splits on the feature.\n\n"
            "**CatBoost (prediction change):** how much the prediction changes, on average, when the feature's "
            "value changes.\n\n"
            "**Caveat:** both are computed on the *training* data and favour continuous features with many possible "
            "split points (e.g. `EstimatedSalary`, `CreditScore`), even when those splits only fit noise. "
            "The two models also distribute credit differently: Random Forest's deep, random trees spread it widely, "
            "while XGBoost's shallow boosted trees concentrate on the strongest splits."
        )

    with t2:
        order = pd.DataFrame({m: t["importance_mean"] for m, t in perm.items()}).mean(axis=1).sort_values().index
        fig = go.Figure()
        for name, table in perm.items():
            fig.add_bar(y=order, x=table.loc[order, "importance_mean"], orientation="h", name=name,
                        marker_color=MODEL_COLORS[name],
                        error_x=dict(type="data", array=table.loc[order, "importance_std"], color=BASELINE, thickness=1),
                        customdata=hover3(table.loc[order, "importance_mean"], table.loc[order, "importance_std"]),
                        hovertemplate="%{y}: %{customdata[0]} ± %{customdata[1]}<extra>%{fullData.name}</extra>")
        fig.add_vline(x=0, line_color=BASELINE, line_width=1)
        fig.update_layout(barmode="group", bargap=0.25, xaxis=dict(title="Drop in test ROC-AUC when shuffled"),
                          legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"))
        plotly(fig, 560)
        st.markdown(
            "**How it works:** take the fitted pipeline, randomly shuffle one *raw* column of the test set (which "
            "breaks its link with the target while keeping its distribution), and measure how much ROC-AUC drops. "
            "Repeat to get a mean and a standard deviation (error bars). Because it shuffles raw columns, "
            "`Geography` and `Gender` are measured as whole features, and it works the same way for every model, "
            "including Logistic Regression.\n\n"
            "**Caveat:** correlated features can share credit. Shuffling one of two redundant features barely hurts "
            "if the model can fall back on the other."
        )

    with t3:
        ranks = pd.DataFrame({f"{m} (built-in)": A.builtin_importance[m].rank(ascending=False).astype(int) for m in TREE_MODELS})
        for m, t in perm.items():
            ranks[f"{m} (permutation)"] = t["importance_mean"].rank(ascending=False).astype(int)
        ranks = ranks.loc[ranks.mean(axis=1).sort_values().index]
        st.markdown("**Rank of each feature by method** (1 = most important)")
        st.dataframe(ranks, width="stretch")
        st.markdown("**Automated insights**")
        bullets(importance_insights(A.builtin_importance, perm))
        lr_np = perm["Logistic Regression"].loc["NumOfProducts", "importance_mean"]
        tree_np = np.mean([perm[m].loc["NumOfProducts", "importance_mean"] for m in TREE_MODELS])
        st.markdown(
            f"**A telling example: `NumOfProducts`.** Churn by number of products in the training data is "
            + format_segments(segment_stats(A.X_train, A.y_train, "NumOfProducts"))
            + f". This is not a straight line, so Logistic Regression (one slope per feature) gains almost nothing "
            f"from it (permutation importance {fmt3(lr_np)}), while the tree models split it into ranges and rely on "
            f"it heavily ({fmt3(tree_np)} on average). This is one concrete reason the tree ensembles rank customers better."
        )


# =============================================================================
# 7. Interactive churn prediction tool
# =============================================================================
def page_predict() -> None:
    page_header(7, "Enter a customer profile and see the churn risk update in real time.")
    what_why_how(
        "Passes the profile through the same fitted pipelines used for evaluation (encoding and scaling "
        "included) and returns <code>predict_proba</code> for <code>Exited = 1</code> from each model.",
        "A probability is more useful than a yes/no label: it lets the bank rank customers and apply a "
        "threshold that reflects its own costs (sidebar strategy).",
        "The gauge shows the probability against the active threshold. The driver chart shows how each input "
        "moves the probability compared with a typical training customer. This explains the <i>model</i>, not the cause of churn.",
    )

    X_train, typical = A.X_train, A.typical

    def num(col, step, is_int=True):
        cast = int if is_int else float
        lo, hi = cast(X_train[col].min()), cast(X_train[col].max())
        return st.number_input(FEATURE_LABELS[col], lo, hi, cast(round(typical[col], 2)), cast(step), key=f"in_{col}",
                               help=f"Training range {lo:,} – {hi:,}. Default = typical customer.")

    with st.container(border=True):
        top = st.columns([3, 1])
        top[0].markdown("##### Customer profile")
        names = list(A.models)
        model_name = top[1].selectbox("Model", names, index=names.index(best_model(A.cv_summary())), key="pred_model",
                                      help="Default: the model selected by cross-validation (highest CV ROC-AUC).")
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.caption("DEMOGRAPHICS")
            age = num("Age", 1)
            gender = st.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(typical["Gender"]), key="in_Gender")
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"],
                                     index=["France", "Germany", "Spain"].index(typical["Geography"]), key="in_Geography")
        with c2:
            st.caption("FINANCIAL PROFILE")
            credit = num("CreditScore", 1)
            balance = num("Balance", 1000, is_int=False)
            salary = num("EstimatedSalary", 1000, is_int=False)
        with c3:
            st.caption("RELATIONSHIP WITH THE BANK")
            tenure = num("Tenure", 1)
            products = st.selectbox("Number of products", [1, 2, 3, 4], index=int(typical["NumOfProducts"]) - 1, key="in_NumOfProducts")
            has_card = st.toggle("Has credit card", value=bool(typical["HasCrCard"]), key="in_HasCrCard")
            active = st.toggle("Active member", value=bool(typical["IsActiveMember"]), key="in_IsActiveMember")

    customer = pd.DataFrame([{
        "CreditScore": credit, "Age": age, "Tenure": tenure, "Balance": balance, "NumOfProducts": products,
        "HasCrCard": int(has_card), "IsActiveMember": int(active), "EstimatedSalary": salary,
        "Geography": geography, "Gender": gender,
    }])[config.FEATURES]

    probs = {name: float(positive_class_proba(m, customer)[0]) for name, m in A.models.items()}
    p, t = probs[model_name], THRESHOLDS[model_name]
    flagged = bool(apply_threshold([p], t)[0])
    base = A.base_rate
    if flagged:
        level, headline = "high", "High churn risk: flagged for retention"
        reason = f"P(churn) = {p:.1%} is at or above the decision threshold of {t:.1%} ({strategy})."
    elif p >= base:
        level, headline = "elevated", "Elevated risk: not flagged"
        reason = f"P(churn) = {p:.1%} is below the threshold ({t:.1%}) but above the average churn rate ({base:.1%})."
    else:
        level, headline = "low", "Low churn risk"
        reason = f"P(churn) = {p:.1%} is below the average churn rate in the training data ({base:.1%})."

    st.subheader("Prediction")
    g, v = st.columns([2, 3], gap="large")
    with g:
        steps = [dict(range=[0, base * 100], color="rgba(27,175,122,0.18)")]
        if t > base:
            steps.append(dict(range=[base * 100, t * 100], color="rgba(237,161,0,0.18)"))
        steps.append(dict(range=[max(t, base) * 100, 100], color="rgba(227,73,72,0.15)"))
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number=dict(suffix="%", valueformat=".1f"),
            title=dict(text=f"P(Exited = 1) · {model_name}", font=dict(size=14)),
            gauge=dict(axis=dict(range=[0, 100], ticksuffix="%"), bar=dict(color=MODEL_COLORS[model_name], thickness=0.3),
                       steps=steps, threshold=dict(line=dict(color="#e34948", width=3), thickness=0.85, value=t * 100)),
        ))
        plotly(gauge, 270)
        st.caption(f"Red line = decision threshold ({t:.1%}). Bands: below average · above average · flagged.")
    with v:
        st.markdown(f'<div class="verdict {level}"><h3>{headline}</h3><p>{reason}</p></div>', unsafe_allow_html=True)
        m = st.columns(2)
        m[0].metric("Churn probability", f"{p:.1%}",
                    help="The model's P(Exited = 1). Compared with the threshold to decide whether to flag. "
                         "Scores are not calibrated probabilities.")
        m[1].metric("vs. average customer", f"{p / base:.2f}×", help=f"Training churn rate {base:.1%}")

        st.markdown("**Probability breakdown: all models**")
        fig = go.Figure()
        fig.add_bar(y=names, x=[1 - probs[n] for n in names], orientation="h", name="P(stay)",
                    marker_color=CLASS_COLORS["Stayed"], text=[f"Stay {1 - probs[n]:.1%}" for n in names],
                    textposition="inside", hovertemplate="%{y}: P(stay) %{x:.1%}<extra></extra>")
        fig.add_bar(y=names, x=[probs[n] for n in names], orientation="h", name="P(churn)",
                    marker_color=CLASS_COLORS["Exited"],
                    text=[f"Churn {probs[n]:.1%}{' · flagged' if probs[n] >= THRESHOLDS[n] else ''}" for n in names],
                    textposition="inside", hovertemplate="%{y}: P(churn) %{x:.1%}<extra></extra>")
        for i, n in enumerate(names):
            fig.add_shape(type="line", x0=1 - THRESHOLDS[n], x1=1 - THRESHOLDS[n], y0=i - 0.42, y1=i + 0.42,
                          line=dict(color="#e34948", width=2))
        fig.update_layout(barmode="stack", xaxis=dict(range=[0, 1], tickformat=".0%"), showlegend=False, bargap=0.3)
        plotly(fig, 200)
        st.caption("Red tick = each model's threshold: the churn segment must reach past it to be flagged.")

    st.subheader("What drives this prediction?")
    exp = explain_prediction(A.models[model_name], customer, typical)
    if exp.empty:
        st.info("This profile equals the typical training customer, so there is nothing to compare against. Change any input above.")
    else:
        exp = exp.assign(label=[f"{FEATURE_LABELS[f]} = {display_value(f, customer[f].iloc[0])}"
                                for f in exp["feature"]]).sort_values("effect")
        fig = go.Figure(go.Bar(
            y=exp["label"], x=exp["effect"] * 100, orientation="h",
            marker_color=[RAISES if e > 0 else LOWERS for e in exp["effect"]],
            text=[f"{e * 100:+.1f} pp" for e in exp["effect"]], textposition="outside",
            hovertemplate="%{y}<br>%{x:+.1f} percentage points vs typical value<extra></extra>",
        ))
        span = max(exp["effect"].abs().max() * 100 * 1.3, 2)
        fig.update_layout(xaxis=dict(title="Change in churn probability vs typical value (percentage points)", range=[-span, span],
                                     zeroline=True, zerolinecolor=BASELINE), yaxis=dict(automargin=True))
        plotly(fig, 120 + 38 * len(exp))
        st.caption(
            "Method: each input is replaced, one at a time, by the typical training value (median or most common) and "
            "the change in the model's probability is recorded. Red raises the risk, green lowers it. Effects are not "
            "additive and describe model behaviour, not causes."
        )

    st.subheader("Business interpretation")
    actions = {
        "high": "**Recommended action:** prioritise for proactive retention outreach. Review the customer's product "
                "fit and recent engagement before making an offer.",
        "elevated": "**Recommended action:** monitor; include in lower-cost retention campaigns if capacity allows.",
        "low": "**Recommended action:** no special retention action indicated; continue standard service.",
    }
    st.markdown(actions[level])
    agreeing = sum(probs[n] >= THRESHOLDS[n] for n in names)
    st.markdown(f"**Model agreement:** {agreeing} of {len(names)} models flag this customer at their current thresholds.")
    st.warning(
        "Probabilities are model scores and are not calibrated. The models were evaluated on one test split and "
        "with cross-validation, but are not validated for real banking decisions. Use this tool for exploration and teaching.",
        icon="⚠️",
    )


# =============================================================================
# 8. Methodology & future roadmap
# =============================================================================
def page_methodology() -> None:
    page_header(8, "The statistics behind the metrics, thresholds and class weights, and where the architecture goes next.")
    what_why_how(
        f"Defined the evaluation metrics for the churn class, added {config.CV_FOLDS}-fold stratified cross-validation, "
        "and built two imbalance levers: class weighting (training) and threshold selection (decision).",
        "Every classifier trades missed churners against false alarms. With ~20% churners, the default 0.5 "
        "threshold and equal weights favour the majority class; both levers must be set without touching the test set.",
        "Use the threshold explorer to see the trade-off curve the optimiser used, and the cost simulator to see "
        "how a business cost ratio would move the threshold.",
    )

    key_terms("Threshold", "Class weighting", "Class imbalance", "Out-of-fold (OOF) prediction", "Gradient boosting",
              "Bagging", "Log-loss", "Overfitting")
    st.subheader("Metric definitions and trade-offs")
    c1, c2 = st.columns(2)
    with c1:
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}")
        st.latex(r"F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}")
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
    with c2:
        st.markdown(
            "- **Precision:** of the customers we flag, how many really churn? Low precision wastes retention budget.\n"
            "- **Recall:** of the customers who churn, how many do we catch? Low recall means lost customers.\n"
            "- **F1:** the harmonic mean, which punishes a model that is good at only one of the two.\n"
            "- **Accuracy:** dominated by the 80% majority class here, so it is supplementary only.\n"
            "- **ROC-AUC:** threshold-free ranking quality; the probability a churner is scored above a non-churner."
        )

    st.subheader("Two levers for class imbalance")
    w = len(A.y_train) / (2 * A.y_train.value_counts().sort_index())
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**1. Class weighting (changes training)**")
        st.latex(r"w_c = \frac{n}{k \cdot n_c}")
        st.markdown(
            f"With n = {len(A.y_train):,} training customers and k = 2 classes: w₀ = **{fmt3(w[0])}** (stayed), "
            f"w₁ = **{fmt3(w[1])}** (churned). Each churner's error counts ~{w[1] / w[0]:.1f}× as much, pushing the "
            f"model to raise churn probabilities. XGBoost expresses the same idea as "
            f"`scale_pos_weight` = #stayed / #churned = **{fmt3(balanced_pos_weight(A.y_train))}**."
        )
    with c2:
        st.markdown("**2. Threshold selection (changes decisions)**")
        st.latex(r"\hat{y} = \mathbb{1}\left[\,P(\text{churn}\mid x) \ge t\,\right]")
        st.markdown(
            "Leaves the model unchanged and moves the cut-off *t*. The CV-optimized strategy picks, per model, the "
            "*t* that maximises F1 on out-of-fold training predictions. Both levers shift the Precision/Recall "
            "balance; combining them can over-correct, so compare settings in Section 3."
        )

    st.subheader("Threshold explorer (training-set cross-validation)")
    c1, c2 = st.columns([1, 3])
    with c1:
        names = list(A.models)
        model_name = st.selectbox("Model", names, index=names.index(best_model(A.cv_summary())), key="meth_model")
        tuned = A.tuned_thresholds[model_name]
        t = st.slider("Threshold", 0.05, 0.95, float(round(tuned, 2)), 0.01, key="meth_t", help=define("Threshold"))
        oof = A.oof_scores[model_name]
        pred = apply_threshold(oof, t)
        tp = int(((pred == 1) & (A.y_train == 1)).sum())
        fp = int(((pred == 1) & (A.y_train == 0)).sum())
        fn = int(((pred == 0) & (A.y_train == 1)).sum())
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        st.metric("CV Precision", fmt3(prec), help=define("Precision"))
        st.metric("CV Recall", fmt3(rec), help=define("Recall"))
        st.metric("CV F1", fmt3(2 * prec * rec / (prec + rec) if prec + rec else 0), help=define("F1"))
    with c2:
        curve = threshold_curve(A.y_train, oof)
        fig = go.Figure()
        for col, color, label in [("precision", "#2a78d6", "Precision"), ("recall", "#eb6834", "Recall"), ("f1", "#1baf7a", "F1")]:
            fig.add_scatter(x=curve["threshold"], y=curve[col], mode="lines", name=label, line=dict(color=color, width=2),
                            customdata=hover3(curve["threshold"], curve[col]),
                            hovertemplate=f"t = %{{customdata[0]}} · {label} %{{customdata[1]}}<extra></extra>")
        fig.add_vline(x=0.5, line_dash="dot", line_color=BASELINE, annotation_text="Default 0.5", annotation_position="top")
        fig.add_vline(x=tuned, line_dash="dash", line_color="#1baf7a", annotation_text=f"F1-optimal {fmt3(tuned)}",
                      annotation_position="bottom right")
        fig.add_vline(x=t, line_color="#e34948", line_width=1.5)
        fig.update_layout(xaxis=dict(title="Decision threshold", range=[0, 1]), yaxis=dict(title="Score", range=[0, 1.02]),
                          legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
        plotly(fig, 400)
    st.markdown(
        f"These curves use **out-of-fold** predictions from the same {config.CV_FOLDS}-fold cross-validation: every "
        "training customer is scored by a pipeline that did not see them, so the curves are out-of-sample, yet the "
        "test set is never used. The *CV-optimized* strategy picks the threshold at the peak of the green F1 curve."
    )

    st.subheader("Cost-sensitive thresholds (what-if)")
    c1, c2, c3 = st.columns(3)
    cost_fn = c1.number_input("Cost of a missed churner", 1.0, 1000.0, 5.0, 1.0, help="Relative units, e.g. lost customer value")
    cost_fp = c2.number_input("Cost of a false alarm", 1.0, 1000.0, 1.0, 1.0, help="Relative units, e.g. cost of a retention offer")
    cost_t = select_cost_threshold(A.y_train, A.oof_scores[model_name], cost_fn, cost_fp)
    c3.metric("Cost-minimising threshold (CV)", fmt3(cost_t), help=f"For {model_name}, on out-of-fold training predictions")
    st.caption(
        "The cost ratio here is a hypothetical input, not a measured business figure. This illustrates how a bank "
        "would turn its own cost estimates into a threshold."
    )

    st.subheader("How gradient boosting learns (animation)")
    c1, c2 = st.columns([5, 2], gap="large")
    with c1:
        st.image(str(boosting_gif()), width="stretch")
    with c2:
        st.markdown(
            "Real gradient boosting on a small **synthetic** churn-vs-age problem, one frame per round:\n\n"
            "1. **Left:** the ensemble's P(churn) (blue) gradually takes the shape of the true risk (dashed).\n"
            "2. **Middle:** the residuals *y − p*, i.e. what the ensemble still gets wrong. The next shallow "
            "tree (orange) is fitted to exactly these errors.\n"
            "3. **Right:** training loss keeps falling, but the loss on **unseen** customers levels off and then "
            "creeps up: extra trees start fitting noise (**overfitting**).\n\n"
            "This is why `n_estimators`, `learning_rate` and `max_depth` are tuned together (Section 4). "
            "XGBoost, LightGBM and CatBoost all follow this loop, with extra regularisation."
        )
    st.caption("Regenerate with `python -m churn.animation`. Synthetic data, for illustration only.")

    st.subheader("Model hyperparameters (documented baselines)")
    rows = [{"Model": "Logistic Regression", "Hyperparameter": "C", "Value": "1.0", "Meaning": HYPERPARAMETERS["C"]}]
    for model, settings in [("Logistic Regression", LOGISTIC_PARAMS), ("Random Forest", FOREST_PARAMS),
                            ("XGBoost", XGBOOST_PARAMS), ("LightGBM", LIGHTGBM_PARAMS), ("CatBoost", CATBOOST_PARAMS)]:
        for k, v in settings.items():
            if k in HYPERPARAMETERS:
                rows.append({"Model": model, "Hyperparameter": k, "Value": str(v), "Meaning": HYPERPARAMETERS[k]})
    params = pd.DataFrame(rows)
    chosen = st.multiselect("Show models", list(MODEL_COLORS), default=list(MODEL_COLORS), key="meth_params",
                            help="Filter the table. Section 4 shows the values selected by GridSearchCV.")
    st.dataframe(params[params["Model"].isin(chosen)], hide_index=True, width="stretch",
                 column_config={"Meaning": st.column_config.Column(width="large"),
                                "Hyperparameter": st.column_config.Column(help=define("Hyperparameter"))})
    st.caption("Technical settings without a modelling effect (n_jobs, thread_count, verbose, file output) are omitted.")

    st.subheader("Architecture: designed for extension")
    st.code(
        '''# churn/models.py: one registry of documented pipelines
def build_models(class_weight=None, y_train=None) -> dict[str, Pipeline]:
    return {"Logistic Regression": ..., "Random Forest": ..., "XGBoost": ..., "LightGBM": ..., "CatBoost": ...}
    # add a new model here; CV, evaluation, importance, report and dashboard pick it up

# churn/tuning.py: GridSearchCV on the training folds, before-vs-after comparison, saved tuned models
search = tune_model("XGBoost", X_train, y_train)            # PARAM_GRIDS["XGBoost"]

# churn/validation.py: stratified k-fold CV on the training set
fold_scores, oof_scores = cross_validate_model(pipeline, X_train, y_train)

# churn/thresholds.py: decision strategies, fitted on out-of-fold training predictions
threshold = select_f1_threshold(y_train, oof_scores)        # or select_cost_threshold(...)

# churn/importance.py + churn/insights.py: importance and automated, evidence-based summaries''',
        language="python",
    )

    st.subheader("Roadmap")
    roadmap = pd.DataFrame([
        ("XGBoost, LightGBM, CatBoost (gradient boosting)", "✅ Implemented", "churn/models.py · documented baseline parameters"),
        ("Stratified k-fold cross-validation", "✅ Implemented", f"{config.CV_FOLDS} folds · CLI report + Section 4"),
        ("Feature importance (built-in + permutation)", "✅ Implemented", "churn/importance.py · Section 5"),
        ("Class weighting", "✅ Implemented", "Sidebar · class_weight / scale_pos_weight"),
        ("Decision-threshold optimisation (F1, CV)", "✅ Implemented", "Sidebar · churn/thresholds.py"),
        ("Cost-sensitive threshold selection", "✅ Implemented (what-if)", "Needs real cost figures from the business"),
        ("Hyperparameter tuning (GridSearchCV)", "✅ Implemented", "Random Forest and XGBoost · Section 4 · outputs/models/"),
        ("Gradient-boosting animation", "✅ Implemented", "churn/animation.py · this section"),
        ("Nested cross-validation for tuning", "🗓️ Planned", "Removes the selection bias of tuned CV scores"),
        ("Tune LightGBM and CatBoost", "🗓️ Planned", "Same GridSearchCV workflow; currently baselines"),
        ("Resampling (SMOTE / undersampling)", "🗓️ Planned", "Must run inside the pipeline on training folds only"),
        ("Probability calibration (CalibratedClassifierCV)", "🗓️ Planned", "Makes scores interpretable as probabilities"),
        ("SHAP explanations", "🗓️ Planned", "Per-customer attributions; associations only"),
        ("Bootstrap confidence intervals on the test set", "🗓️ Planned", "Complements the CV standard deviations"),
    ], columns=["Enhancement", "Status", "Notes"])
    st.dataframe(roadmap, hide_index=True, width="stretch")

    st.subheader("Limitations")
    st.markdown(
        "- Only Random Forest and XGBoost are tuned, over small grids; the other models use baselines.\n"
        "- Tuned CV scores are slightly optimistic (chosen on the same folds); the test set is the fair comparison.\n"
        "- The F1 criterion weights Precision and Recall equally, which may not match real business costs.\n"
        "- A single static dataset of unknown origin and period; patterns may not transfer.\n"
        "- Models and importances capture statistical associations, not causal effects."
    )


# =============================================================================
# 9. The Chief Retention Officer: an interactive machine learning decision story
# =============================================================================


@st.cache_data(show_spinner="Preparing the game with real test-set customers…")
def story_game_html(class_weight: str | None, weighting: str, assets_version: str) -> str:
    """The self-contained HTML/JS game, with this configuration's real model scores baked in.

    assets_version (a hash of game.html and engine.js) is only part of the cache key: editing the game's
    files invalidates the cached page.
    """
    a = get_analysis(class_weight)
    payload = build_payload(a, best=best_model(a.cv_summary()), colors=MODEL_COLORS, weighting=weighting)
    return render_html(payload)


def page_story_game() -> None:
    page_header(9, "Run retention at a commercial bank for five weeks. Every click is a decision; every outcome is real.")
    st.caption(
        "Everything below is interactive and runs instantly in your browser. Customers are real held-out test "
        "customers (fictional names), scored live by every model, and their outcomes are real. Customer values, "
        "incentive costs and success rates are game assumptions. "
        f"Current sidebar weighting: **{weighting_label.split(' (')[0]}**."
    )
    # st.iframe replaces the deprecated st.components.v1.html. The HTML is generated by this project
    # (never user input), as st.iframe requires for HTML with JavaScript.
    # height="content": the iframe grows and shrinks with the game (e.g. one card per row on phones), so there
    # is never a second scrollbar inside the page.
    html = story_game_html(class_weight, weighting_label.split(" (")[0], assets_fingerprint())
    st.iframe(html, height="content")
    with st.expander("How the game works, and what is real"):
        st.markdown(
            "- **Story:** four short chapters introduce churn risk probability, the decision threshold, and the "
            "Precision/Recall trade-off, using this project's own numbers.\n"
            "- **Each week:** five persona cards (David, Sarah, Michael, Emma, Lukas). Each is a real test customer "
            "drawn at random from an archetype (seeded per week, never chosen by score or outcome). Click a card to "
            "talk; give incentives with the buttons; the chat bubbles, risk badges and money tickers react instantly.\n"
            "- **Model switcher:** swap Logistic Regression, Random Forest, XGBoost, LightGBM and CatBoost live: risk "
            "scores, flags and the at-scale panel update.\n"
            "- **Threshold:** customers with risk ≥ the threshold are flagged. *Target all flagged* applies one "
            "incentive to them; the at-scale panel shows the real Precision, Recall, waste and net value of that rule "
            "on all 2,000 test customers.\n"
            "- **Reveal & debrief:** real outcomes, true savings, wasted budget, lost revenue, ROI, a 0–100 success "
            "score against perfect hindsight, every model's own plan, and generated lessons.\n"
            "- **Engine:** money and scoring logic lives in `churn/game_assets/engine.js`; the test suite runs it "
            "under Node and checks it against the Python reference in `churn/game.py`."
        )


# --- Router ------------------------------------------------------------------------
ROUTES = [page_briefing, page_overview, page_pipeline, page_models, page_tuning, page_cv, page_importance, page_predict, page_methodology, page_story_game]
ROUTES[PAGES.index(page)]()
