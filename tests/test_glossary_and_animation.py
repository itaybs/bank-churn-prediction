import numpy as np
import pytest
from PIL import Image

from churn.animation import boosting_history, make_boosting_gif
from churn.glossary import GLOSSARY, HYPERPARAMETERS, METRIC_HELP, define
from churn.models import CATBOOST_PARAMS, FOREST_PARAMS, LIGHTGBM_PARAMS, XGBOOST_PARAMS
from churn.tuning import PARAM_GRIDS
from churn.validation import CV_METRICS

TECHNICAL = {"n_jobs", "thread_count", "verbose", "allow_writing_files"}


def test_every_metric_and_tuned_hyperparameter_is_explained():
    assert set(METRIC_HELP) == set(CV_METRICS)
    for params in (FOREST_PARAMS, XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS):
        assert set(params) - TECHNICAL <= set(HYPERPARAMETERS)
    for grid in PARAM_GRIDS.values():
        assert {k.split("__", 1)[1] for k in grid} <= set(HYPERPARAMETERS)
    assert all(len(text) < 300 for text in {**GLOSSARY, **HYPERPARAMETERS}.values())  # tooltip-sized


def test_define_fails_loudly_on_unknown_terms():
    assert define("Recall").startswith("Of the customers who really churn")
    with pytest.raises(KeyError):
        define("Recal")


def test_boosting_animation_learns_and_shows_overfitting():
    _, _, _, rounds = boosting_history(n_rounds=30)
    train = [r["loss"] for r in rounds]
    val = [r["val_loss"] for r in rounds]
    assert all(b <= a + 1e-12 for a, b in zip(train, train[1:]))  # training loss never increases
    assert val[-1] > min(val)  # unseen-data loss bottoms out before the end
    assert rounds[-1]["tree_fit"] is None and rounds[0]["tree_fit"] is not None


def test_gif_is_animated(tmp_path):
    path = make_boosting_gif(tmp_path / "anim.gif", n_rounds=3, dpi=30)
    with Image.open(path) as im:
        assert im.n_frames >= 4
        assert np.asarray(im.convert("RGB")).std() > 0
