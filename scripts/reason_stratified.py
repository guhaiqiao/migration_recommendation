#!/usr/bin/env python3
"""Reason-stratified diagnosis: how well does FTR rank within each reason class?

Protocol: current locate_migration protocol (4 labeled CSVs, is_migration=1,
per-language stratified 54/11/rest split, tau=730, covered test definition).
For every test query we rank twice: with FTR (R active) and FT (R removed), so
the per-reason delta isolates the marginal value of the reason-match signal R.
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
        source = query["rem_lib"]
        t0 = parse_time(query["time_stamp"])
        reason = (query.get("reason_label") or "").strip()
        return rank_candidates(kb, source, t0, reason, 730, 1 / 3, None, signals)

    stats = defaultdict(lambda: {"n": 0, "cov": 0, "h1": 0, "h3": 0,
                                 "mrr": 0.0, "ndcg": 0.0, "rank_sum": 0,
                                 "h1_ft": 0, "mrr_ft": 0.0})
    for query in test:
        reason = (query.get("reason_label") or "").strip() or "<empty>"
        ranked = rank(query, "FTR")
        pos = next((i for i, t in enumerate(ranked, 1) if t == query["add_lib"]), None)
        s = stats[reason]
        s["n"] += 1
        if pos is None:
            continue
        s["cov"] += 1
        s["h1"] += pos == 1
        s["h3"] += pos <= 3
        s["mrr"] += 1 / pos
        s["ndcg"] += 1 / math.log2(pos + 1) if pos <= 3 else 0.0
        ranked_ft = rank(query, "FT")
        pos_ft = next((i for i, t in enumerate(ranked_ft, 1) if t == query["add_lib"]), None)
        if pos_ft is not None:
            s["h1_ft"] += pos_ft == 1
            s["mrr_ft"] += 1 / pos_ft

    rows = []
    for reason, s in stats.items():
        c = s["cov"] or 1
        rows.append((reason, s["n"], s["cov"], s["h1"] / c, s["h3"] / c,
                     s["mrr"] / c, s["ndcg"] / c,
                     s["h1_ft"] / c, s["mrr_ft"] / c))
    rows.sort(key=lambda r: -r[1])
    tot_n = sum(s["n"] for s in stats.values())
    tot_c = sum(s["cov"] for s in stats.values())
    tot_h1 = sum(s["h1"] for s in stats.values())
    tot_h3 = sum(s["h3"] for s in stats.values())
    tot_mrr = sum(s["mrr"] for s in stats.values())
    tot_ndcg = sum(s["ndcg"] for s in stats.values())
    tot_h1ft = sum(s["h1_ft"] for s in stats.values())
    tot_mrrft = sum(s["mrr_ft"] for s in stats.values())

    print(f"test n={tot_n} covered={tot_c} | FTR Hit@1={tot_h1/tot_c:.3f} "
          f"MRR={tot_mrr/tot_c:.4f} | FT Hit@1={tot_h1ft/tot_c:.3f} "
          f"MRR={tot_mrrft/tot_c:.4f}")
    print(f"{'reason':<16}{'n':>6}{'cov':>6}{'Hit@1':>7}{'Hit@3':>7}{'MRR':>8}"
          f"{'NDCG@3':>8}{'dHit1':>7}{'dMRR':>8}")
    for r in rows:
        print(f"{r[0]:<16}{r[1]:>6}{r[2]:>6}{r[3]:>7.3f}{r[4]:>7.3f}{r[5]:>8.4f}"
              f"{r[6]:>8.4f}{r[3]-r[7]:>+7.3f}{r[5]-r[8]:>+8.4f}")
    print(f"{'ALL':<16}{tot_n:>6}{tot_c:>6}{tot_h1/tot_c:>7.3f}{tot_h3/tot_c:>7.3f}"
          f"{tot_mrr/tot_c:>8.4f}{tot_ndcg/tot_c:>8.4f}"
          f"{(tot_h1-tot_h1ft)/tot_c:>+7.3f}{(tot_mrr-tot_mrrft)/tot_c:>+8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
