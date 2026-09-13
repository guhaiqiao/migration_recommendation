#!/usr/bin/env python3
"""M8: reason-routed R signal.  Queries are split by reason label; on the
validation set each reason stratum decides whether R helps (FTR) or hurts (FT),
and the test set applies the per-stratum routing.  Reported on three scopes:
full test set, reason-annotated subset, and <empty>-reason subset.

Protocol: current locate_migration protocol (4 labeled CSVs, is_migration=1,
per-language stratified 54/11/rest split, tau=730, covered test definition).
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_historical_ranking import (
    build_knowledge_base, language_of, load_instances, parse_time, positive,
    rank_candidates,
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


def main() -> int:
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

    def rank(query, signals):
        return rank_candidates(kb, query["rem_lib"], parse_time(query["time_stamp"]),
                               reason_of(query), 730, 1 / 3, None, signals)

    def stratum_metrics(queries, signals):
        """per-reason covered MRR and counts"""
        agg = defaultdict(lambda: [0, 0.0])  # reason -> [covered, mrr_sum]
        for q in queries:
            ranked = rank(q, signals)
            pos = next((i for i, t in enumerate(ranked, 1) if t == q["add_lib"]), None)
            if pos is not None:
                agg[reason_of(q)][0] += 1
                agg[reason_of(q)][1] += 1 / pos
        return agg

    # 1) decide routing per stratum on the validation set
    val_ftr = stratum_metrics(val, "FTR")
    val_ft = stratum_metrics(val, "FT")
    routing = {}
    print("=== validation routing decision (per stratum, FTR vs FT MRR) ===")
    print(f"{'reason':<16}{'n':>6}{'FTR':>8}{'FT':>8}{'decision':>10}")
    for reason in sorted(set(val_ftr) | set(val_ft)):
        c1, m1 = val_ftr.get(reason, [0, 0.0])
        c2, m2 = val_ft.get(reason, [0, 0.0])
        if reason == "<empty>" or c1 == 0 or c2 == 0:
            routing[reason] = "FT"  # no R available / no evidence -> neutral FT
            decision = "FT(no R)"
        else:
            routing[reason] = "FTR" if m1 / c1 >= m2 / c2 else "FT"
            decision = routing[reason]
        print(f"{reason:<16}{c1:>6}{m1 / max(c1, 1):>8.4f}{m2 / max(c2, 1):>8.4f}"
              f"{decision:>10}")

    # 2) apply on the test set
    def eval_routed(queries):
        cov = h1 = h3 = mrr = ndcg = 0.0
        for q in queries:
            reason = reason_of(q)
            signals = routing.get(reason, "FTR")
            ranked = rank(q, signals)
            pos = next((i for i, t in enumerate(ranked, 1) if t == q["add_lib"]), None)
            if pos is None:
                continue
            cov += 1
            h1 += pos == 1
            h3 += pos <= 3
            mrr += 1 / pos
            ndcg += 1 / math.log2(pos + 1) if pos <= 3 else 0.0
        return cov, h1, h3, mrr, ndcg

    def eval_uniform(queries, signals):
        cov = h1 = h3 = mrr = ndcg = 0.0
        for q in queries:
            ranked = rank(q, signals)
            pos = next((i for i, t in enumerate(ranked, 1) if t == q["add_lib"]), None)
            if pos is None:
                continue
            cov += 1
            h1 += pos == 1
            h3 += pos <= 3
            mrr += 1 / pos
            ndcg += 1 / math.log2(pos + 1) if pos <= 3 else 0.0
        return cov, h1, h3, mrr, ndcg

    print("\n=== test results (covered) ===")
    print(f"{'scope':<14}{'variant':<8}{'cov':>6}{'Hit@1':>8}{'Hit@3':>8}"
          f"{'MRR':>8}{'NDCG@3':>8}")
    scopes = {
        "full": test,
        "annotated": [q for q in test if reason_of(q) != "<empty>"],
        "empty": [q for q in test if reason_of(q) == "<empty>"],
    }
    for scope_name, queries in scopes.items():
        for variant, fn in [("FT", lambda: eval_uniform(queries, "FT")),
                            ("FTR", lambda: eval_uniform(queries, "FTR")),
                            ("routed", lambda: eval_routed(queries))]:
            cov, h1, h3, mrr, ndcg = fn()
            print(f"{scope_name:<14}{variant:<8}{int(cov):>6}{h1 / cov:>8.3f}"
                  f"{h3 / cov:>8.3f}{mrr / cov:>8.4f}{ndcg / cov:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
