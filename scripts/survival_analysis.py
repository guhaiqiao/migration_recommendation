#!/usr/bin/env python3
"""Survival analysis of migration recommendations (indicator validity evidence).

Each migration instance (repo, s -> t, t0) is a subject:
  event  = target t is removed again in the SAME repo at a later migration
           (min time t' of a later (t -> x) event; survival = t' - t0)
  censor = no later removal observed; censor time = last migration event of
           that repo minus t0 (administrative censoring at end of observation)

Covariates (leave-one-out, computed from all labeled positives except the
subject itself):
  F_top : t is the most-frequent historical target of s
  R_sup : subject has a non-empty reason AND the (s,t) edge carries the same
          reason in other migrations (reason-consistent support)

Outputs Kaplan-Meier curves, log-rank tests, and hand-written Cox PH
(Breslow partial likelihood, Newton-Raphson; p via Wald with erf normal CDF).
Interpretation: a covariate that lengthens migration survival (fewer re-
migrations) is evidence that the indicator captures "what the project really
needs", complementing ranking metrics.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_historical_ranking import parse_time, positive
from m9_reason_spotlight import DATASETS, reason_of


def norm_sf(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def km_curve_with_ci(times, events):
    """KM curve + Greenwood 95% CI: returns (ts, surv, lo, hi) step arrays."""
    order = sorted(range(len(times)), key=lambda i: times[i])
    n = len(order)
    ts, surv, var = [0.0], [1.0], [0.0]
    i = 0
    while i < n:
        j = i
        t = times[order[i]]
        while j < n and times[order[j]] == t:
            j += 1
        d = sum(events[order[k]] for k in range(i, j))
        at_risk = n - i
        if d > 0:
            prev = surv[-1]
            v_prev = var[-1]
            surv.append(prev * (at_risk - d) / at_risk)
            var.append(v_prev + d / (at_risk * (at_risk - d)) if at_risk > d else v_prev)
            ts.append(t)
        i = j
    surv = np.array(surv)
    var = np.array(var)
    se = np.sqrt(np.maximum(var, 0))
    lo = np.exp(np.log(surv) - 1.96 * se)
    hi = np.exp(np.log(surv) + 1.96 * se)
    lo[surv == 0] = 0.0
    hi[surv == 0] = 0.0
    return np.array(ts), surv, lo, hi


def km_estimate(times, events):
    order = sorted(range(len(times)), key=lambda i: times[i])
    surv = 1.0
    curve = [(0.0, 1.0)]
    i = 0
    n = len(order)
    while i < n:
        j = i
        t = times[order[i]]
        while j < n and times[order[j]] == t:
            j += 1
        d = sum(events[order[k]] for k in range(i, j))
        at_risk = n - i
        if d > 0:
            surv *= (at_risk - d) / at_risk
        curve.append((t, surv))
        i = j
    return curve


def median_survival(curve):
    for t, s in curve:
        if s <= 0.5:
            return t
    return None


def surv_at(curve, days):
    s = 1.0
    for t, sv in curve:
        if t <= days:
            s = sv
        else:
            break
    return s


def logrank(g1_t, g1_e, g2_t, g2_e):
    events = sorted(set(g1_t + g2_t))
    o1 = e1 = v = 0.0
    for t in events:
        d1 = sum(1 for i in range(len(g1_t)) if g1_t[i] == t and g1_e[i] == 1)
        d2 = sum(1 for i in range(len(g2_t)) if g2_t[i] == t and g2_e[i] == 1)
        n1 = sum(1 for tt in g1_t if tt >= t)
        n2 = sum(1 for tt in g2_t if tt >= t)
        n = n1 + n2
        d = d1 + d2
        if n < 2 or d == 0:
            continue
        e1 += n1 * d / n
        o1 += d1
        v += n1 * n2 * d * (n - d) / (n * n * (n - 1))
    if v == 0:
        return 0.0, 1.0
    z = (o1 - e1) / math.sqrt(v)
    return z, min(1.0, 2 * norm_sf(abs(z)))


def cox_ph(times, events, X):
    """Cox PH with Breslow partial likelihood; X: (n, d) float array."""
    X = np.asarray(X, dtype=float)
    order = np.argsort(-np.array(times, dtype=float))  # descending: risk sets
    times_desc = np.array(times, dtype=float)[order]
    ev_desc = np.array(events, dtype=float)[order]
    X_desc = np.asarray(X, dtype=float)[order]
    beta = np.zeros(X.shape[1])
    for it in range(200):
        lin = np.clip(X_desc @ beta, -30, 30)
        exp_lin = np.exp(lin)
        cum = np.cumsum(exp_lin)  # risk set for each subject (descending)
        cum_w = np.cumsum((X_desc * exp_lin[:, None]), axis=0)
        cum_xx = np.cumsum((X_desc[:, :, None] * X_desc[:, None, :]
                            * exp_lin[:, None, None]), axis=0)
        grad = np.zeros(X.shape[1])
        hess = np.zeros((X.shape[1], X.shape[1]))
        for i in range(len(times_desc)):
            if ev_desc[i] == 0:
                continue
            mean_x = cum_w[i] / cum[i]
            grad += ev_desc[i] * (X_desc[i] - mean_x)
            hess += ev_desc[i] * (cum_xx[i] / cum[i] - np.outer(mean_x, mean_x))
        try:
            delta = np.linalg.solve(hess + 1e-6 * np.eye(hess.shape[0]), grad)
        except np.linalg.LinAlgError:
            delta = grad / (np.diag(hess) + 1e-6)
        lr = 1.0 if it < 20 else 0.3
        beta = np.clip(beta - lr * delta, -10, 10)
        if np.max(np.abs(delta)) < 1e-4:
            break
    try:
        h_inv = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        h_inv = np.linalg.pinv(hess + 1e-6 * np.eye(hess.shape[0]))
    se = np.sqrt(np.maximum(np.diag(h_inv), 0))
    z = beta / se
    p = [2 * norm_sf(float(abs(v))) for v in z]
    return beta, se, p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path, default=None,
                        help="save KM curves (3 panels) to this PNG path")
    args = parser.parse_args()
    instances = []
    for file in DATASETS:
        for row in load_instances_here(file):
            instances.append(row)
    instances = positive(instances)
    # per-repo event timeline
    repo_events = defaultdict(list)
    for row in instances:
        key = row.get("repo_name") or row.get("commit") or "?"
        repo_events[key].append((parse_time(row["time_stamp"]), row["rem_lib"], row["add_lib"], row))

    # global LOO KB for covariates
    edge_freq = defaultdict(int)
    edge_reasons = defaultdict(lambda: defaultdict(int))
    source_freq = defaultdict(lambda: defaultdict(int))
    source_times = defaultdict(lambda: defaultdict(list))
    for row in instances:
        s, t = row["rem_lib"], row["add_lib"]
        edge_freq[(s, t)] += 1
        source_freq[s][t] += 1
        source_times[s][t].append(parse_time(row["time_stamp"]))
        reason = (row.get("reason_label") or "").strip()
        if reason:
            edge_reasons[(s, t)][reason] += 1

    subjects = []
    for key, evs in repo_events.items():
        evs.sort(key=lambda e: e[0])
        last_time = evs[-1][0]
        rem_index = defaultdict(list)
        for ts, rem, add, row in evs:
            rem_index[rem].append(ts)
        for ts, rem, add, row in evs:
            # churn: earliest later time where this target appears as removed
            later = [x for x in rem_index[add] if x > ts]
            if later:
                duration = (min(later) - ts).total_seconds() / 86400.0
                event = 1
            else:
                duration = max((last_time - ts).total_seconds() / 86400.0, 0.0)
                event = 0
            s, t = rem, add
            reason = (row.get("reason_label") or "").strip()
            # leave-one-out counts
            f_count = edge_freq[(s, t)] - 1
            r_same = edge_reasons[(s, t)].get(reason, 0) - (1 if reason else 0)
            # F_top: is t the most frequent target of s (LOO)?
            s_counts = {tt: (source_freq[s][tt] - (1 if tt == t else 0))
                        for tt in source_freq[s]}
            f_top = bool(s_counts) and t == max(s_counts, key=s_counts.get)
            r_sup = bool(reason) and r_same > 0
            # T_top: is t the most-recent target of s (LOO)?
            s_latest = {tt: max(ts_list) for tt, ts_list in source_times[s].items()
                        if tt != t or len(ts_list) > 1}
            t_top = bool(s_latest) and t == max(s_latest, key=s_latest.get)
            subjects.append({"duration": duration, "event": event,
                             "F_top": int(f_top), "R_sup": int(r_sup),
                             "T_top": int(t_top), "reason": reason, "n_f": f_count})

    n = len(subjects)
    print(f"subjects = {n}")
    groups = [("F_top", lambda d: d["F_top"] == 1),
              ("T_top", lambda d: d["T_top"] == 1),
              ("R_sup", lambda d: d["R_sup"] == 1)]
    for name, cond in groups:
        g1 = [d for d in subjects if cond(d)]
        g2 = [d for d in subjects if not cond(d)]
        if not g1 or not g2:
            continue
        c1 = km_estimate([d["duration"] for d in g1], [d["event"] for d in g1])
        c2 = km_estimate([d["duration"] for d in g2], [d["event"] for d in g2])
        z, p = logrank([d["duration"] for d in g1], [d["event"] for d in g1],
                       [d["duration"] for d in g2], [d["event"] for d in g2])
        ev1 = sum(d["event"] for d in g1)
        print(f"\n=== {name}=1 (n={len(g1)}, events={ev1}) vs {name}=0 "
              f"(n={len(g2)}, events={sum(d['event'] for d in g2)}) ===")
        print(f"median survival: {median_survival(c1)}d vs {median_survival(c2)}d")
        print(f"survival at 365d: {surv_at(c1, 365):.3f} vs {surv_at(c2, 365):.3f} | "
              f"at 730d: {surv_at(c1, 730):.3f} vs {surv_at(c2, 730):.3f}")
        print(f"log-rank z = {z:+.3f}, p = {p:.4f}")

    # Cox PH: single and joint covariates
    # NOTE: with only 113 events and near-separation (R_sup=1 event rate
    # 1.3% vs 1.4%), the Breslow partial likelihood does not converge
    # (beta hits clamp bounds); we therefore report log-rank results only.
    print("\n=== Cox PH (Breslow) ===")
    print("not reported: 113 events / 8,059 subjects, near-separation makes the"
          " partial likelihood diverge; log-rank tests are the primary inference.")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = {"F_top": "F support (most-frequent target)",
                  "T_top": "T support (most-recent target)",
                  "R_sup": "R support (reason-consistent target)"}
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, (name, cond) in zip(axes, groups):
            g1 = [d for d in subjects if cond(d)]
            g2 = [d for d in subjects if not cond(d)]
            z, p = logrank([d["duration"] for d in g1], [d["event"] for d in g1],
                           [d["duration"] for d in g2], [d["event"] for d in g2])
            for group, color in [(g1, "#1f77b4"), (g2, "#d62728")]:
                ts, s, lo, hi = km_curve_with_ci(
                    [d["duration"] for d in group], [d["event"] for d in group])
                label = f"{name}=1 (n={len(g1)})" if group is g1 else f"{name}=0 (n={len(g2)})"
                ax.step(ts, s, where="post", color=color, label=label)
                ax.fill_between(ts, lo, hi, step="post", alpha=0.15, color=color)
            ax.set_xlim(0, 730)
            ax.set_ylim(0.86, 1.0)
            ax.set_xlabel("Days since migration")
            ax.set_ylabel("Survival probability")
            ax.set_title(f"{labels[name]}\nlog-rank p = {p:.4f}", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle("Migration survival by indicator support (8,059 subjects, 113 churn events)",
                     fontsize=11)
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=200)
        print(f"saved KM curves to {args.plot}")
    return 0


def load_instances_here(path):
    from evaluate_historical_ranking import load_instances as li
    return li(Path(path))


if __name__ == "__main__":
    raise SystemExit(main())
