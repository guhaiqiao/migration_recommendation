#!/usr/bin/env python3
"""Scenario design: subsets where F is expected to weaken and T/R strengthen.

All subset criteria use ONLY query-observable features (train KB statistics,
reason label, F counts) — never the test ground truth — same methodology as the
tiebreak subset.

Scenarios (pooled test set, per-language gate tables as before):
  direction_shift  source's most-FREQUENT target differs from its most-RECENT
                   target -> the crowd wisdom F points at the old mainstream
  f_flat           >=3 candidates and the F-normalised top-1/top-2 gap < 0.1
                   -> F barely discriminates
  reason_active    reason in the functional set (Functionality/BugIssue/
                   Security/Simplification) AND >=1 candidate edge carries it
                   -> R is actually activated (non-trivial)
  f_flat+reason    intersection: F weak AND R strong
Report Hit@1/MRR for Rnd/F/T/R/FT/FTR/FTR* per scenario.
"""
import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_historical_ranking import (
    build_knowledge_base, language_of, load_instances, minmax_normalise,
    parse_time, positive, rank_candidates,
)
from m9_reason_spotlight import DATASETS, reason_of

VARIANTS = ["Rnd", "F", "T", "R", "FT", "FTR", "FTR*"]
FUNCTIONAL = {"Functionality", "BugIssue", "Security", "Simplification"}


def learn_routing(val, kb):
    def agg(signals):
        out = defaultdict(lambda: [0, 0.0])
        for q in val:
            ranked = rank_candidates(kb, q["rem_lib"], parse_time(q["time_stamp"]),
                                     reason_of(q), 730, 1 / 3, None, signals)
            pos = next((i for i, t in enumerate(ranked, 1) if t == q["add_lib"]), None)
            if pos is not None:
                out[reason_of(q)][0] += 1
                out[reason_of(q)][1] += 1 / pos
        return out

    ftr, ft = agg("FTR"), agg("FT")
    routing = {}
    for r in set(ftr) | set(ft):
        c1, m1 = ftr.get(r, [0, 0.0])
        c2, m2 = ft.get(r, [0, 0.0])
        routing[r] = ("FTR" if r != "<empty>" and c1 and c2 and m1 / c1 >= m2 / c2
                      else "FT")
    return routing


def f_counts(query, kb):
    edges = kb.get(query["rem_lib"])
    if not edges:
        return None
    return {t: edges[t]["count"] for t in edges}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps-f", type=float, default=0.1, help="F-normalised gap threshold")
    args = parser.parse_args()

    instances = []
    for f in DATASETS:
        for row in load_instances(Path(f)):
            row["_dataset"] = f.split("/")[-1]
            instances.append(row)
    instances = positive(instances)
    by_lang = defaultdict(list)
    for row in instances:
        by_lang[language_of(row["_dataset"])].append(row)

    all_test = []
    routing_pool = {}
    kb_pool = {}
    for lang, rows in by_lang.items():
        if not rows:
            continue
        rows.sort(key=lambda r: parse_time(r["time_stamp"]))
        n_tr, n_va = int(len(rows) * 0.54), int(len(rows) * 0.11)
        train, val, test = rows[:n_tr], rows[n_tr:n_tr + n_va], rows[n_tr + n_va:]
        kb = build_knowledge_base(train)
        routing = learn_routing(val, kb)
        kb_pool[lang] = kb
        routing_pool[lang] = routing
        for q in test:
            q["_lang"] = lang
            all_test.append(q)

    def ranked(q, variant):
        kb = kb_pool[q["_lang"]]
        routing = routing_pool[q["_lang"]]
        if variant == "FTR*":
            sig = routing.get(reason_of(q), "FTR")
            return rank_candidates(kb, q["rem_lib"], parse_time(q["time_stamp"]),
                                   reason_of(q), 730, 1 / 3, None, sig)
        if variant == "Rnd":
            edges = kb.get(q["rem_lib"])
            if not edges:
                return []
            cands = list(edges.keys())
            rng = random.Random(42)
            rng.shuffle(cands)
            return cands
        return rank_candidates(kb, q["rem_lib"], parse_time(q["time_stamp"]),
                               reason_of(q), 730, 1 / 3, None, variant)

    def in_scenario(q, name):
        kb = kb_pool[q["_lang"]]
        edges = kb.get(q["rem_lib"])
        if not edges:
            return False
        counts = {t: e["count"] for t, e in edges.items()}
        times = {t: max(e["times"]) for t, e in edges.items()}
        most_freq = max(counts, key=counts.get)
        most_recent = max(times, key=times.get)
        reason = reason_of(q)
        if name == "direction_shift":
            return most_freq != most_recent and len(counts) >= 2
        if name == "f_flat":
            if len(counts) < 3:
                return False
            fn = minmax_normalise([float(c) for c in counts.values()])
            ranked_f = sorted(fn, reverse=True)
            return ranked_f[0] - ranked_f[1] < args.eps_f
        if name == "reason_active":
            if reason not in FUNCTIONAL:
                return False
            return any(any(r for r, _ in e["reasons"].items() if r) for e in edges.values())
        if name == "f_flat_reason":
            return in_scenario(q, "f_flat") and in_scenario(q, "reason_active")
        return False

    def metrics(queries, variant):
        cov = h1 = h3 = mrr = 0.0
        for q in queries:
            rk = ranked(q, variant)
            pos = next((i for i, t in enumerate(rk, 1) if t == q["add_lib"]), None)
            if pos is None:
                continue
            cov += 1
            h1 += pos == 1
            h3 += pos <= 3
            mrr += 1 / pos
        if not cov:
            return 0, float("nan"), float("nan"), float("nan")
        return cov, h1 / cov, h3 / cov, mrr / cov

    scenarios = ["full", "direction_shift", "f_flat", "reason_active", "f_flat_reason"]
    print(f"{'scenario':<18}{'n':>6}  " + "  ".join(f"{v:>15}" for v in VARIANTS))
    for name in scenarios:
        queries = all_test if name == "full" else [q for q in all_test if in_scenario(q, name)]
        cells = []
        n_cov = None
        for v in VARIANTS:
            cov, h1, h3, mrr = metrics(queries, v)
            if v == "FTR":
                n_cov = cov
            cells.append(f"{h1:.3f}/{h3:.3f}/{mrr:.4f}" if cov else f"{'--':>15}")
        print(f"{name:<18}{n_cov:>6}  " + "  ".join(cells))
    print("cells are Hit@1/Hit@3/MRR (covered); Rnd=random (seed 42); FTR* uses per-language gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
