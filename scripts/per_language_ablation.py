#!/usr/bin/env python3
"""Per-language ablation of F/T/R and the gated reason signal R* (FTR*).

For EACH language separately:
  - stratified time split 54/11/35 inside the language;
  - gate table g(reason) learned on that language's validation set
    (FTR vs FT per reason stratum); empty-reason stratum always gated off;
  - variants: F, T, R, FT, FR, TR, FTR (baseline), FTR* (gated R);
  - three scopes: full covered, reason-annotated, tiebreak
    (>=4 candidates and top-1/top-2 FTR score gap < eps, eps=0.05).
Each cell prints Hit@1/MRR.
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

VARIANTS = ["Rnd", "F", "T", "R", "FT", "FR", "TR", "FTR", "FTR*"]


def scores_f(query, kb):
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
        rs = matched / total if total else 0.0
        base = (f_norm[i] + t_norm[i]) / 2
        out[t] = (1 - 1 / 3) * base + rs / 3 if reason != "<empty>" else base
    return out


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, default=0.05)
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

    header = (f"{'lang':<9}{'scope':<10}{'n':>6}  " +
              "  ".join(f"{v:>11}" for v in VARIANTS))
    print(header)
    pooled = {"full": [], "annotated": [], "tiebreak": []}
    pooled_routing = {}
    for lang in ["python", "java", "js", "c"]:
        rows = by_lang[lang]
        if not rows:
            continue
        rows.sort(key=lambda r: parse_time(r["time_stamp"]))
        n_tr, n_va = int(len(rows) * 0.54), int(len(rows) * 0.11)
        train, val, test = rows[:n_tr], rows[n_tr:n_tr + n_va], rows[n_tr + n_va:]
        kb = build_knowledge_base(train)
        routing = learn_routing(val, kb)

        def ranked(q, variant, kb=kb, routing=routing):
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
            cov = h1 = mrr = 0.0
            for q in queries:
                rk = ranked(q, variant)
                pos = next((i for i, t in enumerate(rk, 1) if t == q["add_lib"]), None)
                if pos is None:
                    continue
                cov += 1
                h1 += pos == 1
                mrr += 1 / pos
            return int(cov), h1 / cov if cov else float("nan"), mrr / cov if cov else float("nan")

        full = test
        annotated = [q for q in test if reason_of(q) != "<empty>"]
        tie = []
        for q in test:
            sc = scores_f(q, kb)
            if len(sc) < 4:
                continue
            ranked_s = sorted(sc.items(), key=lambda it: -it[1])
            if ranked_s[0][1] - ranked_s[1][1] < args.eps:
                tie.append(q)

        for scope_name, queries in [("full", full), ("annotated", annotated),
                                    ("tiebreak", tie)]:
            pooled[scope_name].append((queries, ranked, metrics))
            cells = []
            n_cov = None
            for v in VARIANTS:
                cov, h1, mrr = metrics(queries, v)
                if v == "FTR":
                    n_cov = cov
                cells.append(f"{h1:.3f}/{mrr:.4f}" if cov else f"{'--':>11}")
            print(f"{lang:<9}{scope_name:<10}{n_cov:>6}  " + "  ".join(cells))
    # pooled row: each query uses its own language's gate table
    for scope_name, batches in pooled.items():
        cells = []
        n_cov = None
        for v in VARIANTS:
            cov = h1 = mrr = 0.0
            for queries, ranked, _ in batches:
                c, h, m = _m(queries, v, ranked)
                cov += c
                h1 += h
                mrr += m
            if v == "FTR":
                n_cov = cov
            cells.append(f"{h1 / cov:.3f}/{mrr / cov:.4f}" if cov else f"{'--':>11}")
        print(f"{'pooled':<9}{scope_name:<10}{n_cov:>6}  " + "  ".join(cells))
    print("cells are Hit@1/MRR; Rnd = random shuffle of candidates (seed 42); "
          "FTR* = FTR with reason-gated R (per-language gate table)")
    return 0


def _m(queries, variant, ranked):
    cov = h1 = mrr = 0.0
    for q in queries:
        rk = ranked(q, variant)
        pos = next((i for i, t in enumerate(rk, 1) if t == q["add_lib"]), None)
        if pos is None:
            continue
        cov += 1
        h1 += pos == 1
        mrr += 1 / pos
    return int(cov), h1, mrr


if __name__ == "__main__":
    raise SystemExit(main())
