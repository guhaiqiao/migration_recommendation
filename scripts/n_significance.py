#!/usr/bin/env python3
"""Paired bootstrap + McNemar significance tests for FTR vs FTRN (M6 negative evidence).

Follows experiments.md section 10: MRR delta via 3,000 paired bootstrap
resamples (95% CI), Hit@1 via exact McNemar on discordant pairs.
"""
import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_historical_ranking import (
    build_knowledge_base, language_of, load_instances, parse_time, positive,
)
from m6_momentum_negative import DATASETS, graph_stats, rank_m6


def binom_tail(n: int, b: int) -> float:
    """Exact two-sided McNemar p-value: discordant pairs split b vs c."""
    # p = P(Binomial(n, .5) >= b) * 2, n = b + c discordant count
    tail = 0.0
    for k in range(b, n + 1):
        tail += math.comb(n, k) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.25, help="churn weight for FTRN")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boot", type=int, default=3000)
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

    def ranks(queries, signals, alpha):
        out = {}
        for query in queries:
            source = query["rem_lib"]
            t0 = parse_time(query["time_stamp"])
            reason = (query.get("reason_label") or "").strip()
            ranked = rank_m6(kb, rev, churn, inflow, source, t0, reason,
                             730, 1 / 3, signals, 180, alpha)
            if query["add_lib"] in ranked:
                out[id(query)] = next(i for i, t in enumerate(ranked, 1)
                                      if t == query["add_lib"])
        return out

    # validation check first: does alpha survive model selection on val?
    vr_ftr = ranks(val, "FTR", 0.0)
    vr_n = ranks(val, "FTRN", args.alpha)
    val_mrr_ftr = sum(1 / r for r in vr_ftr.values()) / len(vr_ftr)
    val_mrr_n = sum(1 / r for r in vr_n.values()) / len(vr_n)
    print(f"val MRR: FTR {val_mrr_ftr:.4f} | FTRN(a={args.alpha}) {val_mrr_n:.4f}")

    tr_ftr = ranks(test, "FTR", 0.0)
    tr_n = ranks(test, "FTRN", args.alpha)
    common = [q for q in tr_ftr if q in tr_n]
    diffs = [1 / tr_n[q] - 1 / tr_ftr[q] for q in common]
    h1_ftr = sum(tr_ftr[q] == 1 for q in common)
    h1_n = sum(tr_n[q] == 1 for q in common)
    b = sum(tr_ftr[q] > 1 and tr_n[q] == 1 for q in common)  # FTRN wins
    c = sum(tr_ftr[q] == 1 and tr_n[q] > 1 for q in common)  # FTR wins
    disc = b + c

    rng = random.Random(args.seed)
    means = []
    n = len(diffs)
    for _ in range(args.boot):
        means.append(sum(rng.choices(diffs, k=n)) / n)
    means.sort()
    lo, hi = means[int(0.025 * args.boot)], means[int(0.975 * args.boot)]
    mean_delta = sum(diffs) / n

    print(f"test common n={n} | MRR FTR {sum(1/tr_ftr[q] for q in common)/n:.4f} "
          f"vs FTRN {sum(1/tr_n[q] for q in common)/n:.4f}")
    print(f"MRR delta = {mean_delta:+.4f}  bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"Hit@1: FTR {h1_ftr/n:.4f} vs FTRN {h1_n/n:.4f} | "
          f"discordant b(FTRN wins)={b} c(FTR wins)={c} | "
          f"McNemar p = {binom_tail(disc, max(b, c)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
