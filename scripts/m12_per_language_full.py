#!/usr/bin/env python3
"""Per-language full-view metrics with Random and Fa baselines (cells:
Hit@1/Hit@3/MRR/NDCG@3).  Same protocol as m11_full_frequency.py; gate tables
learned per language on that language's validation set."""
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_historical_ranking import (
    build_knowledge_base, language_of, load_instances, parse_time, positive,
    rank_candidates,
)
from m9_reason_spotlight import DATASETS, load_rows_with_non_migrations, reason_of
from m11_full_frequency import learn_routing, rank_fa

VARIANTS = ["Rnd", "Fa", "F", "T", "R", "FT", "FR", "TR", "FTR", "FTR*"]


def main() -> int:
    rows_all = load_rows_with_non_migrations()
    positives = positive(rows_all)
    by_lang_pos = defaultdict(list)
    by_lang_all = defaultdict(list)
    for row in positives:
        by_lang_pos[language_of(row["_dataset"])].append(row)
    for row in rows_all:
        by_lang_all[language_of(row["_dataset"])].append(row)

    per_lang = {}
    for lang in ["python", "java", "js", "c"]:
        pos_rows = by_lang_pos.get(lang, [])
        if not pos_rows:
            continue
        pos_rows.sort(key=lambda r: parse_time(r["time_stamp"]))
        n_tr, n_va = int(len(pos_rows) * 0.54), int(len(pos_rows) * 0.11)
        train, val, test = pos_rows[:n_tr], pos_rows[n_tr:n_tr + n_va], pos_rows[n_tr + n_va:]
        cutoff = parse_time(train[-1]["time_stamp"])
        all_lang = sorted(by_lang_all.get(lang, []), key=lambda r: parse_time(r["time_stamp"]))
        kb = build_knowledge_base(train)
        kb_all = build_knowledge_base([r for r in all_lang if parse_time(r["time_stamp"]) < cutoff])
        routing = learn_routing(val, kb)
        per_lang[lang] = (kb, kb_all, routing, test)

    def rank(q, variant):
        kb, kb_all, routing, _ = per_lang[q["_lang"]]
        if variant == "Fa":
            return rank_fa(kb_all, q["rem_lib"])
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

    def metrics(queries, variant):
        cov = h1 = h3 = mrr = ndcg = 0.0
        for q in queries:
            rk = rank(q, variant)
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

    all_test = []
    for lang, (_, _, _, test) in per_lang.items():
        for q in test:
            q["_lang"] = lang
            all_test.append(q)

    print(f"{'lang':<9}{'n':>6}  " + "  ".join(f"{v:>19}" for v in VARIANTS))
    for lang in ["python", "java", "js", "c"]:
        queries = [q for q in all_test if q["_lang"] == lang]
        cells = []
        n_cov = None
        for v in VARIANTS:
            cov, h1, h3, mrr, ndcg = metrics(queries, v)
            if v == "FTR":
                n_cov = cov
            cells.append(f"{h1:.3f}/{h3:.3f}/{mrr:.4f}/{ndcg:.4f}" if cov else f"{'--':>19}")
        print(f"{lang:<9}{n_cov:>6}  " + "  ".join(cells))
    # pooled row
    cells = []
    n_cov = None
    for v in VARIANTS:
        cov, h1, h3, mrr, ndcg = metrics(all_test, v)
        if v == "FTR":
            n_cov = cov
        cells.append(f"{h1:.3f}/{h3:.3f}/{mrr:.4f}/{ndcg:.4f}" if cov else f"{'--':>19}")
    print(f"{'pooled':<9}{n_cov:>6}  " + "  ".join(cells))
    print("cells Hit@1/Hit@3/MRR/NDCG@3; Fa = frequency over ALL replacements (incl. non-migrations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
