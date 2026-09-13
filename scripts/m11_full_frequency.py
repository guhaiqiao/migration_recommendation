#!/usr/bin/env python3
"""F vs F_all: does including NON-migration replacements in the frequency count
help or hurt?

F   = frequency counted from migration-labeled positives only (current signal)
Fa  = frequency counted from ALL dependency replacements in the training period
      (including is_migration=0 rows: version upgrades, refactorings, etc.)

Both use the same protocol (per-language 54/11/35 split on positives, train KB
built from rows strictly before the train cutoff -> no leakage).  The test set
and all scenario definitions are unchanged.
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
from m9_reason_spotlight import DATASETS, load_rows_with_non_migrations, reason_of

VARIANTS = ["Rnd", "Fa", "F", "T", "R", "FT", "FR", "TR", "FTR", "FTR*"]
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


def rank_fa(kb_all, source):
    edges = kb_all.get(source)
    if not edges:
        return []
    targets = sorted(edges.keys(), key=lambda t: (-edges[t]["count"], t))
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps-f", type=float, default=0.1)
    args = parser.parse_args()

    rows_all = load_rows_with_non_migrations()
    positives = positive(rows_all)
    by_lang_pos = defaultdict(list)
    by_lang_all = defaultdict(list)
    for row in positives:
        by_lang_pos[language_of(row["_dataset"])].append(row)
    for row in rows_all:
        by_lang_all[language_of(row["_dataset"])].append(row)

    all_test = []
    for lang in ["python", "java", "js", "c"]:
        pos_rows = by_lang_pos.get(lang, [])
        if not pos_rows:
            continue
        pos_rows.sort(key=lambda r: parse_time(r["time_stamp"]))
        n_tr, n_va = int(len(pos_rows) * 0.54), int(len(pos_rows) * 0.11)
        train, val, test = pos_rows[:n_tr], pos_rows[n_tr:n_tr + n_va], pos_rows[n_tr + n_va:]
        cutoff = parse_time(train[-1]["time_stamp"])
        all_rows_lang = sorted(by_lang_all.get(lang, []), key=lambda r: parse_time(r["time_stamp"]))
        kb = build_knowledge_base(train)
        kb_all = build_knowledge_base([r for r in all_rows_lang
                                       if parse_time(r["time_stamp"]) < cutoff])
        routing = learn_routing(val, kb)
        n_extra = sum(len(e) for e in kb_all.values()) - sum(len(e) for e in kb.values())
        for q in test:
            q["_lang"] = lang
            q["_kb"] = kb
            q["_kb_all"] = kb_all
            q["_routing"] = routing
            all_test.append(q)
        print(f"{lang}: train pos {len(train)}, kb edges (mig) "
              f"{sum(len(e) for e in kb.values())}, kb_all extra edges {n_extra}")

    def ranked(q, variant):
        kb = q["_kb"]
        routing = q["_routing"]
        if variant == "Fa":
            return rank_fa(q["_kb_all"], q["rem_lib"])
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
        kb = q["_kb"]
        edges = kb.get(q["rem_lib"])
        if not edges:
            return False
        counts = {t: e["count"] for t, e in edges.items()}
        times = {t: max(e["times"]) for t, e in edges.items()}
        if name == "direction_shift":
            return len(counts) >= 2 and max(counts, key=counts.get) != max(times, key=times.get)
        if name == "f_flat":
            if len(counts) < 3:
                return False
            fn = minmax_normalise([float(c) for c in counts.values()])
            ranked_f = sorted(fn, reverse=True)
            return ranked_f[0] - ranked_f[1] < args.eps_f
        if name == "reason_active":
            reason = reason_of(q)
            if reason not in FUNCTIONAL:
                return False
            return any(any(r for r, _ in e["reasons"].items() if r) for e in edges.values())
        return False

    def metrics(queries, variant):
        cov = h1 = h3 = mrr = ndcg = 0.0
        for q in queries:
            rk = ranked(q, variant)
            pos = next((i for i, t in enumerate(rk, 1) if t == q["add_lib"]), None)
            if pos is None:
                continue
            cov += 1
            h1 += pos == 1
            h3 += pos <= 3
            mrr += 1 / pos
            ndcg += 1 / math.log2(pos + 1) if pos <= 3 else 0.0
        if not cov:
            return 0, float("nan"), float("nan"), float("nan"), float("nan")
        return cov, h1 / cov, h3 / cov, mrr / cov, ndcg / cov

    scenarios = ["full", "direction_shift", "f_flat", "reason_active"]
    print(f"\n{'scenario':<18}{'n':>6}  " + "  ".join(f"{v:>19}" for v in VARIANTS))
    for name in scenarios:
        queries = all_test if name == "full" else [q for q in all_test if in_scenario(q, name)]
        cells = []
        n_cov = None
        for v in VARIANTS:
            cov, h1, h3, mrr, ndcg = metrics(queries, v)
            if v == "FTR":
                n_cov = cov
            cells.append(f"{h1:.3f}/{h3:.3f}/{mrr:.4f}/{ndcg:.4f}" if cov else f"{'--':>19}")
        print(f"{name:<18}{n_cov:>6}  " + "  ".join(cells))
    print("cells Hit@1/Hit@3/MRR/NDCG@3; Fa = frequency over ALL replacements (incl. non-migrations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
