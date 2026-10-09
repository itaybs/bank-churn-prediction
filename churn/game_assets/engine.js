/* The Chief Retention Officer: game engine (pure functions, no DOM).
 *
 * Mirrors churn/game.py so the browser game and the tested Python engine agree:
 * tests/test_story_game.py runs this file under Node and compares results.
 *
 * Money rules (expected values):
 *   targeted churner   -> savings = success * value, lost = (1 - success) * value
 *   targeted loyal     -> wasted = cost                (a false positive)
 *   untargeted churner -> lost = value                 (a false negative)
 *   untargeted loyal   -> nothing
 */
(function (root) {
  "use strict";

  const NONE = "none";

  function incentiveById(econ, id) {
    if (id === NONE || id == null) return { id: NONE, cost: 0, success: 0 };
    const inc = econ.incentives.find((i) => i.id === id);
    if (!inc) throw new Error("Unknown incentive " + id);
    return inc;
  }

  function outcome(decision, churned, value, econ) {
    const inc = incentiveById(econ, decision);
    const targeted = inc.cost > 0;
    return {
      spend: inc.cost,
      true_savings: targeted && churned ? inc.success * value : 0,
      wasted: targeted && !churned ? inc.cost : 0,
      lost_revenue: churned ? (1 - inc.success) * value : 0,
    };
  }

  function summarise(decisions, churned, values, econ) {
    const t = { spend: 0, true_savings: 0, wasted: 0, lost_revenue: 0 };
    decisions.forEach((d, i) => {
      const o = outcome(d, churned[i], values[i], econ);
      for (const k in t) t[k] += o[k];
    });
    t.net = t.true_savings - t.spend;
    t.roi = t.spend ? t.net / t.spend : 0;
    return t;
  }

  function spend(decisions, econ) {
    return decisions.reduce((s, d) => s + incentiveById(econ, d).cost, 0);
  }

  /* Expected net value of a plan if the customers churn with probabilities `weights`. */
  function expectedNet(decisions, weights, values, econ) {
    return decisions.reduce((s, d, i) => {
      const inc = incentiveById(econ, d);
      return s + weights[i] * inc.success * values[i] - inc.cost;
    }, 0);
  }

  /* Exhaustive search (few customers): maximise sum(weight * success * value - cost) within budget.
   * Ties go to the cheaper plan, then to the first plan found (same order as Python's itertools.product). */
  function bestPlan(weights, values, econ, budget) {
    const options = [NONE].concat(econ.incentives.map((i) => i.id));
    const n = values.length;
    let best = null, bestGain = -Infinity, bestCost = Infinity;
    const plan = new Array(n);
    (function rec(k, cost) {
      if (cost > budget + 1e-9) return;
      if (k === n) {
        const gain = Math.round(expectedNet(plan, weights, values, econ) * 1e6) / 1e6;
        if (gain > bestGain || (gain === bestGain && cost < bestCost)) {
          best = plan.slice(); bestGain = gain; bestCost = cost;
        }
        return;
      }
      for (const o of options) {
        plan[k] = o;
        rec(k + 1, cost + incentiveById(econ, o).cost);
      }
    })(0, 0);
    return best;
  }

  function successScore(net, bestNet) {
    if (bestNet <= 0) return net >= 0 ? 100 : 0;
    return Math.round(100 * Math.min(Math.max(net / bestNet, 0), 1));
  }

  const BADGES = [
    [80, "🏆", "Retention Champion", "You captured most of the value a perfectly informed manager could have saved."],
    [60, "🥈", "Savvy Strategist", "Solid, data-driven decisions with a healthy return."],
    [30, "📈", "Promising Manager", "Some good calls, but money was left on the table or spent on loyal customers."],
    [0, "🪙", "Cautious Starter", "No money lost, but most of the value you could have saved slipped away."],
  ];
  const BUDGET_BURNER = ["💸", "Budget Burner", "The budget did more harm than good. Check the risk scores and try again."];

  function badge(score, net) {
    if (net < 0) return BUDGET_BURNER;
    return BADGES.find((b) => score >= b[0]).slice(1);
  }

  /* At-scale statistics for one model at threshold t (nearest grid point), flagging p >= t and giving
   * every flagged customer the same incentive. grid: {thresholds, tp, fp, fn, tn, v_tp, v_fn}. */
  function scaleStats(grid, t, incentive) {
    let i = 0, bestD = Infinity;
    grid.thresholds.forEach((g, k) => { const d = Math.abs(g - t); if (d < bestD - 1e-12) { bestD = d; i = k; } });
    const tp = grid.tp[i], fp = grid.fp[i], fn = grid.fn[i], tn = grid.tn[i];
    const flagged = tp + fp;
    const savings = incentive.success * grid.v_tp[i];
    const spendAll = incentive.cost * flagged;
    return {
      threshold: grid.thresholds[i], tp, fp, fn, tn, flagged,
      precision: flagged ? tp / flagged : 0,
      recall: tp + fn ? tp / (tp + fn) : 0,
      savings, spend: spendAll, wasted: incentive.cost * fp,
      lost: grid.v_fn[i] + (1 - incentive.success) * grid.v_tp[i],
      net: savings - spendAll,
    };
  }

  const Engine = { NONE, incentiveById, outcome, summarise, spend, expectedNet, bestPlan, successScore, badge, scaleStats };
  if (typeof module !== "undefined" && module.exports) module.exports = Engine;
  else root.Engine = Engine;
})(typeof window !== "undefined" ? window : this);
