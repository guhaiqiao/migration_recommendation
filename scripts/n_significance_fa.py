#!/usr/bin/env python3
"""Paired bootstrap + McNemar for Fa vs F on the two counter-intuitive subsets
(f_flat, reason_active) from m11_full_frequency.py."""
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
from m11_full_frequency import rank_fa, FUNCTIONAL


def binom_tail(n, b):
    tail = 0.0
    for k in range(b, n + 1):
        tail += math.comb(n, k) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def main() -> int:
    rows_all = load_rows_with_non_migrations()
    positives = positive(rows_all)
    by_lang_pos = defaultdict(list)
    by_lang_all = defaultdict(list)
    for row in positives:
        by_lang_pos[language_of(row["_dataset"])].append(row)
    for row in rows_all:
        by_lang_all[language_of(row["_dataset"])].append(row)

    tests = {}
    for lang in ["python", "java", "js", "c"]:
        pos_rows = by_lang_pos.get(lang, [])
        if not pos_rows:
            continue
        pos_rows.sort(key=lambda r: parse_time(r["time_stamp"]))
        n_tr = int(len(pos_rows) * 0.54)
        train, test = pos_rows[:n_tr], pos_rows[n_tr:]
        cutoff = parse_time(train[-1]["time_stamp"])
        all_lang = sorted(by_lang_all.get(lang, []), key=lambda r: parse_time(r["time_stamp"]))
        kb = build_knowledge_base(train)
        kb_all = build_knowledge_base([r for r in all_lang if parse_time(r["time_stamp"]) < cutoff])
        tests[lang] = (kb, kb_all, test)

    def rank(q, variant):
        kb, kb_all, _ = tests[q["_lang"]]
        if variant == "Fa":
            return rank_fa(kb_all, q["rem_lib"])
        return rank_candidates(kb, q["rem_lib"], parse_time(q["time_stamp"]),
                               reason_of(q), 730, 1 / 3, None, variant)

    def in_scenario(q, name):
        kb, _, _ = tests[q["_lang"]]
        edges = kb.get(q["rem_lib"])
        if not edges:
            return False
        counts = {t: e["count"] for t, e in edges.items()}
        if name == "f_flat":
            if len(counts) < 3:
                return False
            fn = minmax_normalise([float(c) for c in counts.values()])
            ranked_f = sorted(fn, reverse=True)
            return ranked_f[0] - ranked_f[1] < 0.1
        if name == "reason_active":
            if reason_of(q) not in FUNCTIONAL:
                return False
            return any(any(r for r, _ in e["reasons"].items() if r) for e in edges.values())
        return False

    all_test = [q for lang, (_, _, test) in tests.items()
                for q in test if (q.__setitem__("_lang", lang) is None)]

    rng = random.Random(42)
    for name in ["f_flat", "reason_active"]:
        queries = [q for q in all_test if in_scenario(q, name)]
        pairs = []
        h1a = h1f = 0
        for q in queries:
            ra = rank(q, "Fa")
            rf = rank(q, "F")
            pa = next((i for i, t in enumerate(ra, 1) if t == q["add_lib"]), None)
            pf = next((i for i, t in enumerate(rf, 1) if t == q["add_lib"]), None)
            if pf is None:
                continue
            pairs.append((1 / pf, 1 / pa if pa else 0))
            h1f += pf == 1
            h1a += pa == 1
        n = len(pairs)
        diffs = [b - a for a, b in pairs]
        boot = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(3000))
        lo, hi = boot[75], boot[2924]
        disc_b = sum(1 for q in queries
                     if (next((i for i, t in enumerate(rank(q, "Fa"), 1) if t == q["add_lib"]), None) == 1)
                     != (next((i for i, t in enumerate(rank(q, "F"), 1) if t == q["add_lib"]), None) == 1))
        fa_wins = sum(1 for q in queries
                      if (next((i for i, t in enumerate(rank(q, "Fa"), 1) if t == q["add_lib"]), None) == 1)
                      and (next((i for i, t in enumerate(rank(q, "F"), 1) if t == q["add_lib"]), None) != 1))
        f_wins = sum(1 for q in queries
                     if (next((i for i, t in enumerate(rank(q, "F"), 1) if t == q["add_lib"]), None) == 1)
                     and (next((i for i, t in enumerate(rank(q, "Fa"), 1) if t == q["add_lib"]), None) != 1))
        p_mcn = binom_tail(disc_b, max(fa_wins, f_wins))
        print(f"{name}: n={n} | Hit@1 Fa {h1a / n:.3f} vs F {h1f / n:.3f} | "
              f"MRR delta {sum(diffs) / n:+.4f} 95% CI [{lo:+.4f},{hi:+.4f}] | "
              f"McNemar discordant Fa-wins={fa_wins} F-wins={f_wins} p={p_mcn:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
