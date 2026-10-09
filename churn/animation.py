"""An animated GIF that shows how gradient boosting learns, round by round.

The animation runs real gradient boosting for binary classification (log-loss)
on a small synthetic one-feature problem: churn probability against age, with
a hump-shaped true risk, similar in spirit to the bank data. Each round:

1. compute the residuals y - p (the negative gradient of the log-loss);
2. fit a shallow regression tree (depth 2, at least 40 customers per leaf) to them;
3. set each leaf to a Newton step, sum(residual) / sum(p * (1 - p));
4. add the tree, shrunk by the learning rate, to the log-odds.

This is the update used by gradient-boosting libraries such as XGBoost,
LightGBM and CatBoost (which add regularisation and faster split finding).
A held-out validation sample shows when extra trees stop helping (overfitting).
The data are synthetic and for illustration only.

Regenerate with:  python -m churn.animation
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.tree import DecisionTreeRegressor

from churn import config

GIF_PATH = config.PROJECT_ROOT / "assets" / "gradient_boosting.gif"
MODEL, TREE, TRUE, POINTS, TEXT, MUTED, GRID = "#2a78d6", "#eb6834", "#8a8984", "#9085e9", "#0b0b0b", "#52514e", "#e4e3df"


def _true_probability(age: np.ndarray) -> np.ndarray:
    return 0.08 + 0.62 * np.exp(-((age - 54) / 9) ** 2)


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


MIN_LEAF = 40


def _log_loss(y, p):
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def boosting_history(n_rounds: int = 30, learning_rate: float = 0.3, n: int = 600, seed: int = config.RANDOM_STATE):
    """Run gradient boosting and record every round. Returns (x, y, grid, rounds)."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(18, 85, n))
    y = (rng.uniform(size=n) < _true_probability(x)).astype(float)
    x_val = rng.uniform(18, 85, n)
    y_val = (rng.uniform(size=n) < _true_probability(x_val)).astype(float)
    grid = np.linspace(18, 85, 300)

    base = np.log(y.mean() / (1 - y.mean()))  # round 0: every customer gets the overall churn rate
    f, f_grid, f_val = np.full(n, base), np.full(grid.size, base), np.full(n, base)
    rounds = []
    for m in range(n_rounds + 1):
        p = _sigmoid(f)
        residual = y - p
        record = dict(p_grid=_sigmoid(f_grid), residual=residual, tree_fit=None,
                      loss=_log_loss(y, p), val_loss=_log_loss(y_val, _sigmoid(f_val)))
        rounds.append(record)
        if m == n_rounds:
            break
        tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=MIN_LEAF, random_state=seed).fit(x[:, None], residual)
        leaves = tree.apply(x[:, None])
        gamma = {leaf: residual[leaves == leaf].sum() / max((p * (1 - p))[leaves == leaf].sum(), 1e-9)
                 for leaf in np.unique(leaves)}
        record["tree_fit"] = tree.predict(grid[:, None])
        update = [np.vectorize(gamma.get)(tree.apply(v[:, None])) for v in (x, grid, x_val)]
        f, f_grid, f_val = (a + learning_rate * u for a, u in zip((f, f_grid, f_val), update))
    return x, y, grid, rounds


def make_boosting_gif(path: Path = GIF_PATH, n_rounds: int = 30, learning_rate: float = 0.3, fps: int = 3,
                      dpi: int = 80) -> Path:
    x, y, grid, rounds = boosting_history(n_rounds, learning_rate)
    jitter = np.random.default_rng(1).uniform(-0.03, 0.03, len(y))
    losses = [r["loss"] for r in rounds]
    val_losses = [r["val_loss"] for r in rounds]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.3), dpi=dpi, gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    frames = list(range(len(rounds))) + [len(rounds) - 1] * 4  # hold the final frame

    def draw(i):
        r = rounds[i]
        for ax in (ax1, ax2, ax3):
            ax.clear()
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)

        ax1.scatter(x, y + jitter, s=6, color=POINTS, alpha=0.35, linewidths=0, label="Customers (0 = stayed, 1 = churned)")
        ax1.plot(grid, _true_probability(grid), color=TRUE, linestyle="--", linewidth=1.5, label="True churn probability")
        ax1.plot(grid, r["p_grid"], color=MODEL, linewidth=2.5, label="Model P(churn)")
        ax1.set_ylim(-0.08, 1.08)
        ax1.set_xlabel("Age", color=TEXT, fontsize=9)
        ax1.set_title(f"1 · The ensemble after {i} tree{'s' if i != 1 else ''}", color=TEXT, fontsize=11, loc="left")
        ax1.legend(loc="upper left", fontsize=7, frameon=False)

        ax2.scatter(x, r["residual"], s=6, color=POINTS, alpha=0.35, linewidths=0, label="Residual y − p")
        ax2.axhline(0, color=TRUE, linewidth=1)
        if r.get("tree_fit") is not None:
            ax2.plot(grid, r["tree_fit"], color=TREE, linewidth=2.5, label=f"Tree {i + 1} (depth 2) fitted to residuals")
            ax2.set_title(f"2 · What is still wrong → tree {i + 1} learns it", color=TEXT, fontsize=11, loc="left")
        else:
            ax2.set_title("2 · Residuals after the last round", color=TEXT, fontsize=11, loc="left")
        ax2.set_ylim(-1.05, 1.05)
        ax2.set_xlabel("Age", color=TEXT, fontsize=9)
        ax2.legend(loc="upper left", fontsize=7, frameon=False)

        for series, color, label in [(losses, MODEL, "Training"), (val_losses, TREE, "Validation (unseen)")]:
            ax3.plot(range(len(series)), series, color=GRID, linewidth=2)
            ax3.plot(range(i + 1), series[: i + 1], color=color, linewidth=2.5, label=f"{label} {series[i]:.3f}")
            ax3.scatter([i], [series[i]], s=40, color=color, zorder=3, edgecolors="white", linewidths=1.5)
        ax3.set_xlabel("Boosting round (trees added)", color=TEXT, fontsize=9)
        ax3.set_ylabel("Log-loss (lower is better)", color=TEXT, fontsize=9)
        ax3.set_title("3 · Training vs unseen-data loss", color=TEXT, fontsize=11, loc="left")
        ax3.legend(loc="upper right", fontsize=7, frameon=False)
        fig.suptitle(f"Gradient boosting: each new tree corrects the errors of the trees before it "
                     f"(learning rate {learning_rate})", color=TEXT, fontsize=12)
        fig.tight_layout()

    animation = FuncAnimation(fig, draw, frames=frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(f"Wrote {make_boosting_gif()}")
