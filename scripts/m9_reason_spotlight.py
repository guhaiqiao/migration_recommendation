#!/usr/bin/env python3
"""M9: test-set designs that make the reason effect visible.

All subset definitions use only query-observable features (reason label,
candidate count, FTR scores) or the train KB — never the test ground truth —
so no subset is cherry-picked on the true target.

Views:
  micro          full covered test set (baseline averaging, dilutes rare reasons)
  macro          equal weight per reason stratum (macro-average of Hit@1/MRR)
  r-present      queries with a non-empty reason label (R is turned on)
  r-nontrivial   r-present AND >=1 candidate edge in the KB carries this reason
                 (R actually differentiates: the pure-effect subset)
  tiebreak       covered queries with >=4 candidates AND top-1/top-2 FTR score
                 gap < eps (the regime where an extra signal can flip rank 1)

Each view compares FT / FTR / routed(M8) with per-stratum routing decided on
the validation set (same as m8_reason_routing.py).
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_historical_ranking import (
    build_knowledge_base, language_of, load_instances, minmax_normalise,
    parse_time, positive, rank_candidates,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASETS = [
    str(DATA_DIR / "python_positive.csv"),
    str(DATA_DIR / "java_positive.csv"),
    str(DATA_DIR / "js_positive.csv"),
    str(DATA_DIR / "c_positive.csv"),
]


def reason_of(row):
    return (row.get("reason_label") or "").strip() or "<empty>"


def load_rows_with_non_migrations():
    rows = []
    for f in DATASETS:
        for row in load_instances(Path(f)):
            row["_dataset"] = f.split("/")[-1]
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, default=0.05, help="tiebreak score-gap threshold")
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

    def scores(query):
        """FTR raw scores per candidate (recompute of rank_candidates internals)."""
        edges = kb.get(query["rem_lib"])
        if not edges:
            return {}
        targets = list(edges.keys())
        f_norm = minmax_normalise([float(edges[t]["count"]) for t in targets])
        t_norm = minmax_normalise(
            [math.exp(-max((parse_time(query["time_stamp"])
                            - max(edges[t]["times"])).days, 0) / 730) for t in targets])
        reason = reason_of(query)
        out = {}
        for i, t in enumerate(targets):
            edge = edges[t]
            matched = sum(c for r, c in edge["reasons"].items() if r and r == reason)
            total = sum(c for r, c in edge["reasons"].items() if r)
            r_s = matched / total if total else 0.0
            base = (f_norm[i] + t_norm[i]) / 2
            out[t] = (1 - 1 / 3) * base + r_s / 3 if reason != "<empty>" else base
        return out

    def rank(query, signals):
        return rank_candidates(kb, query["rem_lib"], parse_time(query["time_stamp"]),
                               reason_of(query), 730, 1 / 3, None, signals)

    def stratum_metrics(queries, signals):
        agg = defaultdict(lambda: [0, 0.0])
        for q in queries:
            ranked = rank(q, signals)
            pos = next((i for i, t in enumerate(ranked, 1) if t == q["add_lib"]), None)
            if pos is not None:
                agg[reason_of(q)][0] += 1
                agg[reason_of(q)][1] += 1 / pos
        return agg

    val_ftr = stratum_metrics(val, "FTR")
    val_ft = stratum_metrics(val, "FT")
    routing = {}
    for reason in set(val_ftr) | set(val_ft):
        c1, m1 = val_ftr.get(reason, [0, 0.0])
        c2, m2 = val_ft.get(reason, [0, 0.0])
        if reason == "<empty>" or c1 == 0 or c2 == 0:
            routing[reason] = "FT"
        else:
            routing[reason] = "FTR" if m1 / c1 >= m2 / c2 else "FT"

    def subset_of(queries, kind):
        if kind == "micro":
            return queries
        if kind == "r-present":
            return [q for q in queries if reason_of(q) != "<empty>"]
        if kind == "r-nontrivial":
            out = []
            for q in queries:
                reason = reason_of(q)
                if reason == "<empty>":
                    continue
                edges = kb.get(q["rem_lib"])
                if edges and any(any(r for r, _ in e["reasons"].items() if r) for e in edges.values()):
                    out.append(q)
            return out
        if kind == "tiebreak":
            out = []
            for q in queries:
                sc = scores(q)
                if len(sc) < 4:
                    continue
                ranked = sorted(sc.items(), key=lambda it: -it[1])
                if ranked[0][1] - ranked[1][1] < args.eps:
                    out.append(q)
            return out
        raise ValueError(kind)

    def eval_view(queries, signals, routed=False):
        cov = h1 = h3 = mrr = ndcg = 0.0
        for q in queries:
            sig = routing.get(reason_of(q), "FTR") if routed else signals
            ranked = rank(q, sig)
            pos = next((i for i, t in enumerate(ranked, 1) if t == q["add_lib"]), None)
            if pos is None:
                continue
            cov += 1
            h1 += pos == 1
            h3 += pos <= 3
            mrr += 1 / pos
            ndcg += 1 / math.log2(pos + 1) if pos <= 3 else 0.0
        return cov, h1, h3, mrr, ndcg

    print(f"=== M9 views (eps={args.eps}) | train {len(train)} val {len(val)} test {len(test)} ===")
    print(f"{'view':<14}{'variant':<8}{'cov':>6}{'Hit@1':>8}{'Hit@3':>8}"
          f"{'MRR':>8}{'NDCG@3':>8}")
    for kind in ["micro", "r-present", "r-nontrivial", "tiebreak"]:
        queries = subset_of(test, kind)
        if not queries:
            continue
        for variant, routed in [("FT", False), ("FTR", False), ("routed", True)]:
            cov, h1, h3, mrr, ndcg = eval_view(queries, variant, routed)
            print(f"{kind:<14}{variant:<8}{int(cov):>6}{h1 / cov:>8.3f}"
                  f"{h3 / cov:>8.3f}{mrr / cov:>8.4f}{ndcg / cov:>8.4f}")

    # macro-average: equal weight per reason stratum (excluding <empty>)
    print("\n=== macro-average over reason strata (non-empty, covered) ===")
    agg_ftr = stratum_metrics(test, "FTR")
    agg_ft = stratum_metrics(test, "FT")
    agg_rt = defaultdict(lambda: [0, 0.0])
    for q in test:
        ranked = rank(q, routing.get(reason_of(q), "FTR"))
        pos = next((i for i, t in enumerate(ranked, 1) if t == q["add_lib"]), None)
        if pos is not None:
            agg_rt[reason_of(q)][0] += 1
            agg_rt[reason_of(q)][1] += 1 / pos
    for label, agg in [("FT", agg_ft), ("FTR", agg_ftr), ("routed", agg_rt)]:
        strata = [r for r in agg if r != "<empty>"]
        macro_mrr = sum(agg[r][1] / max(agg[r][0], 1) for r in strata) / len(strata)
        print(f"{label:<8} strata={len(strata):>3} macro-MRR={macro_mrr:.4f}")

    # error repair counts on the r-present subset (routed vs FTR)
    def repaired_routed(queries):
        n_fix = n_break = 0
        for q in queries:
            rb = rank(q, "FTR")
            rr = rank(q, routing.get(reason_of(q), "FTR"))
            pb = next((i for i, t in enumerate(rb, 1) if t == q["add_lib"]), None)
            pr = next((i for i, t in enumerate(rr, 1) if t == q["add_lib"]), None)
            if pr == 1 and pb != 1:
                n_fix += 1
            elif pr != 1 and pb == 1:
                n_break += 1
        return n_fix, n_break
    rp = subset_of(test, "r-present")
    fix, brk = repaired_routed(rp)
    print(f"\nr-present errors repaired by routing vs FTR: fixed {fix} broken {brk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
