#!/usr/bin/env python3
"""M6 ablation: momentum M + negative evidence N on top of the F/T/R history signals.

Protocol: same as the "current" protocol in protocol_combo_ablation.py
  - four labeled CSVs filtered by is_migration=1 (python/java/js/c)
  - per-language stratified split: 54% train / 11% val / rest test (time-ordered)
  - tau=730 (fixed)
  - "covered" test definition: true target must be among the KB candidates
    (no leakage: KB is built from train rows only, all events < t0)

New signals (both computed from the train KB only, so no future leakage):

  M  momentum: share of the source->target events falling in a recent window
     (t0 - window, t0].  M = count_recent / count.  Captures targets whose
     migrations are *accelerating* vs steadily frequent (complementary to F).

  N  negative evidence: N = 1 - minmax(rev + alpha * churn), where
       rev(s,t)  = number of opposite-direction migrations t -> s in the KB
       churn(t)  = total migrations *away from* t (t appears as source)
     Targets that are themselves being abandoned, or that migrate back to s,
     are down-weighted.

Fusion rule matches rank_candidates: exact "FTR" keeps the weighted formula
w_R*R + (1-w_R)*(F+T)/2; any other subset averages the selected components.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_historical_ranking import (
    build_knowledge_base, language_of, load_instances, minmax_normalise,
    parse_time, positive,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASETS = [
    str(DATA_DIR / "python_positive.csv"),
    str(DATA_DIR / "java_positive.csv"),
    str(DATA_DIR / "js_positive.csv"),
    str(DATA_DIR / "c_positive.csv"),
]


def graph_stats(kb: dict):
    """rev[(s,t)] = opposite-direction count; churn[t] = migrations away from t;
    inflow[t] = migrations into t."""
    rev = {}
    churn = defaultdict(int)
    inflow = defaultdict(int)
    for source, edges in kb.items():
        for target, edge in edges.items():
            rev[(source, target)] = kb.get(target, {}).get(source, {}).get("count", 0)
            churn[target] += edge["count"]
            inflow[source] += edge["count"]
    return rev, churn, inflow


def rank_m6(kb: dict, rev: dict, churn: dict, inflow: dict, source: str, t0: datetime,
            reason: str, tau: int = 730, w_reason: float = 1 / 3,
            signals: str = "FTRMN", window: int = 180, alpha: float = 0.5) -> list[str]:
    edges = kb.get(source)
    if not edges:
        return []
    targets = list(edges.keys())
    f_norm = minmax_normalise([float(edges[t]["count"]) for t in targets])
    t_norm = minmax_normalise(
        [math.exp(-max((t0 - max(edges[t]["times"])).days, 0) / tau) for t in targets])
    r_scores = []
    for target in targets:
        edge = edges[target]
        matched = sum(c for r, c in edge["reasons"].items() if r and r == reason)
        total = sum(c for r, c in edge["reasons"].items() if r)
        r_scores.append(matched / total if total else 0.0)
    m_raw = []
    for target in targets:
        recent = sum(1 for ts in edges[target]["times"] if (t0 - ts).days <= window)
        m_raw.append(math.log1p(recent) - math.log1p(edges[target]["count"] - recent))
    m_norm = minmax_normalise(m_raw)
    n_raw = []
    for target in targets:
        count = edges[target]["count"]
        rev_count = rev.get((source, target), 0)
        rev_ratio = rev_count / (rev_count + count + 1)
        churn_ratio = churn.get(target, 0) / (churn.get(target, 0) + inflow.get(target, 0) + 1)
        n_raw.append(rev_ratio + alpha * churn_ratio)
    n_scores = [1.0 - v for v in minmax_normalise(n_raw)]
    comps = {"F": f_norm, "T": t_norm, "R": r_scores, "M": m_norm, "N": n_scores}
    selected = [c for c in signals.upper() if c in comps]
    items = []
    for idx, target in enumerate(targets):
        if signals.upper() == "FTR" and reason:
            base = (f_norm[idx] + t_norm[idx]) / 2
            score = (1 - w_reason) * base + w_reason * r_scores[idx]
        elif selected:
            score = sum(comps[c][idx] for c in selected) / len(selected)
        else:
            score = 0.0
        items.append({"target": target, "S": score, "count": edges[target]["count"]})
    items.sort(key=lambda item: (-item["S"], -item["count"], item["target"]))
    return [item["target"] for item in items]


def rank_candidates_ftr(kb, source, t0, reason, tau=730, signals="FTR"):
    """Baseline wrapper for combo comparison (identical to current protocol)."""
    edges = kb.get(source)
    if not edges:
        return []
    targets = list(edges.keys())
    f_norm = minmax_normalise([float(edges[t]["count"]) for t in targets])
    t_norm = minmax_normalise(
        [math.exp(-max((t0 - max(edges[t]["times"])).days, 0) / tau) for t in targets])
    r_scores = []
    for target in targets:
        edge = edges[target]
        matched = sum(c for r, c in edge["reasons"].items() if r and r == reason)
        total = sum(c for r, c in edge["reasons"].items() if r)
        r_scores.append(matched / total if total else 0.0)
    comps = {"F": f_norm, "T": t_norm, "R": r_scores}
    selected = [c for c in signals.upper() if c in comps]
    items = []
    for idx, target in enumerate(targets):
        if signals.upper() == "FTR" and reason:
            base = (f_norm[idx] + t_norm[idx]) / 2
            score = (1 - 1 / 3) * base + r_scores[idx] / 3
        elif selected:
            score = sum(comps[c][idx] for c in selected) / len(selected)
        else:
            score = 0.0
        items.append((target, score, edges[target]["count"]))
    items.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [target for target, _, _ in items]


def spearman(xs, ys):
    """Rank correlation without scipy; ties get average ranks."""
    def ranks(vals):
        order = sorted(range(len(vals)), key=vals.__getitem__)
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    n = len(xs)
    rx, ry = ranks(list(xs)), ranks(list(ys))
    mx, my = (n + 1) / 2.0, (n + 1) / 2.0
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n))
                    * sum((ry[i] - my) ** 2 for i in range(n)))
    return num / den if den > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=180, help="M recent window in days")
    parser.add_argument("--alpha", type=float, default=0.5, help="churn weight in N")
    parser.add_argument("--combos", default="FTR,M,N,FTRM,FTRN,FTRMN,TRM")
    args = parser.parse_args()

    instances = []
    for file in DATASETS:
        for row in load_instances(Path(file)):
            row["_dataset"] = file.split("/")[-1]
            instances.append(row)
    instances = positive(instances)
    by_lang = defaultdict(list)
    for row in instances:
        by_lang[language_of(row["_dataset"])].append(row)
    train, val, test = [], [], []
    for lang, rows in by_lang.items():
        rows.sort(key=lambda row: parse_time(row["time_stamp"]))
        n_tr, n_va = int(len(rows) * 0.54), int(len(rows) * 0.11)
        train.extend(rows[:n_tr])
        val.extend(rows[n_tr:n_tr + n_va])
        test.extend(rows[n_tr + n_va:])
    kb = build_knowledge_base(train)
    rev, churn, inflow = graph_stats(kb)

    def covered(queries, signals):
        out = []
        for query in queries:
            source = query["rem_lib"]
            t0 = parse_time(query["time_stamp"])
            reason = (query.get("reason_label") or "").strip()
            ranked = rank_m6(kb, rev, churn, inflow, source, t0, reason,
                             730, 1 / 3, signals, args.window, args.alpha)
            if query["add_lib"] in ranked:
                out.append((query, ranked))
        return out

    def metrics(queries, signals):
        n, h1, h3, mrr, ndcg = 0, 0.0, 0.0, 0.0, 0.0
        for query, ranked in covered(queries, signals):
            rank = next(i for i, t in enumerate(ranked, 1) if t == query["add_lib"])
            n += 1
            h1 += rank == 1
            h3 += rank <= 3
            mrr += 1 / rank
            ndcg += 1 / math.log2(rank + 1) if rank <= 3 else 0.0
        return n, h1, h3, mrr, ndcg

    # collinearity check: raw signal values over val candidate pairs
    f_raw, t_raw, m_raw, n_raw = [], [], [], []
    for query in val:
        source = query["rem_lib"]
        t0 = parse_time(query["time_stamp"])
        edges = kb.get(source)
        if not edges:
            continue
        for target, edge in edges.items():
            count = edge["count"]
            recent = sum(1 for ts in edge["times"] if (t0 - ts).days <= args.window)
            rev_count = rev.get((source, target), 0)
            f_raw.append(count)
            t_raw.append((t0 - max(edge["times"])).days)
            m_raw.append(math.log1p(recent) - math.log1p(count - recent))
            n_raw.append(rev_count / (rev_count + count + 1)
                         + args.alpha * churn.get(target, 0)
                         / (churn.get(target, 0) + inflow.get(target, 0) + 1))

    print(f"=== M6 | current protocol, 4-language positives | train {len(train)} "
          f"val {len(val)} test {len(test)} | window {args.window}d alpha {args.alpha} ===")
    print(f"corr(F,M)={spearman(f_raw, m_raw):+.3f}  corr(F,N)={spearman(f_raw, n_raw):+.3f}  "
          f"corr(T,M)={spearman(t_raw, m_raw):+.3f}  "
          f"corr(F,T)={spearman(f_raw, t_raw):+.3f}")
    print(f"{'combo':<8}{'val_MRR':>9}{'test_n':>7}{'Hit@1':>8}{'Hit@3':>8}{'MRR':>8}{'NDCG@3':>8}")
    for signals in [c.strip().upper() for c in args.combos.split(",")]:
        nv, _, _, vmrr, _ = metrics(val, signals)
        n_test, h1, h3, mrr, ndcg = metrics(test, signals)
        print(f"{signals:<8}{vmrr/nv:>9.4f}{n_test:>7}"
              f"{h1/n_test:>8.3f}{h3/n_test:>8.3f}{mrr/n_test:>8.4f}{ndcg/n_test:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
