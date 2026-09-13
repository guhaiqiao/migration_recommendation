#!/usr/bin/env python3
"""M7 candidate generation expansion: transitive + co-migration + deprecation.

Protocol: current locate_migration protocol (four labeled CSVs filtered by
is_migration=1, per-language stratified split 54/11/rest, tau=730, history
KB from train only -> no future leakage).

Expansion sources (all computed from train rows, i.e. strictly before t0):
  transit    s -> m -> t paths of length 2 in the KB graph, score
             lambda_t * (S(s->m) + S(m->t)) / 2 with S the FTR score
  co-mig     targets y of sources s' that were removed in the same commit as
             s (same repo + commit_sha); score lambda_c * count/max_count
  deprecate  targets named after "deprecated ... use X" patterns in the
             commit messages of s; score lambda_d * count/max_count

Baseline M0 ranks with direct history edges only (FTR weighted formula).
Metrics: covered Hit@1/3, MRR, NDCG@3, NDCG@10; all-query variants; and a
cold-start subset report (queries with no history edge: recovery rate,
Top-10 hit, NDCG@10).
"""
import argparse
import math
import re
import sys
from collections import Counter, defaultdict
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

DEP_RE = re.compile(
    r"(?:deprecated|obsoleted?)\s*(?:in\s*(?:favor|favour)\s*of|by|with|use\s+)?"
    r"['\"]?([A-Za-z][A-Za-z0-9_.-]*)", re.I)


def score_edges(kb: dict, source: str, t0: datetime, reason: str,
                tau: int = 730, w_reason: float = 1 / 3) -> dict:
    """FTR weighted score per target for the direct edges of one source."""
    edges = kb.get(source)
    if not edges:
        return {}
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
    out = {}
    for idx, target in enumerate(targets):
        base = (f_norm[idx] + t_norm[idx]) / 2
        out[target] = (1 - w_reason) * base + w_reason * r_scores[idx] if reason else base
    return out


def build_aux(train_rows: list[dict]):
    """co_map[source] = Counter of targets from same-commit co-removed sources;
    dep_map[source] = Counter of targets named in deprecation messages."""
    co_map = defaultdict(Counter)
    dep_map = defaultdict(Counter)
    commits = defaultdict(list)
    for row in train_rows:
        repo = row.get("repo_name") or ""
        sha = row.get("commit_sha") or row.get("commit") or ""
        commits[f"{repo}|{sha}"].append((row["rem_lib"], row["add_lib"]))
        msg = row.get("commit_message") or row.get("message") or ""
        for target in DEP_RE.findall(msg):
            dep_map[row["rem_lib"]][target.lower()] += 1
    for pairs in commits.values():
        if len(pairs) < 2:
            continue
        for s, x in pairs:
            for s2, y in pairs:
                if s2 != s:
                    co_map[s][y] += 1
    return co_map, dep_map


def expand_rank(kb: dict, co_map: dict, dep_map: dict, source: str, t0: datetime,
                reason: str, modes: str, l_transit: float, l_co: float,
                l_dep: float, strategy: str = "append") -> list[tuple]:
    """Return [(target, score, origin, count)] ranked; origin in {direct,
    transit, co, dep}.  append: direct candidates first (unchanged order),
    then expanded ones by score;  mix: single score-sorted list."""
    direct = score_edges(kb, source, t0, reason)
    edge_counts = {t: kb[source][t]["count"] for t in kb.get(source, {})}
    best = {}  # target -> [score, origin, count]
    for t, s in direct.items():
        best[t] = [s, "direct", edge_counts.get(t, 0)]
    if "t" in modes:
        for mid in kb.get(source, {}):
            if mid == source:
                continue
            s_mid = direct.get(mid, 0.0)
            for t, s_mt in score_edges(kb, mid, t0, reason).items():
                if t == source:
                    continue
                sc = l_transit * (s_mid + s_mt) / 2
                if t not in best or sc > best[t][0]:
                    best[t] = [sc, "transit", 0]
    if "c" in modes and source in co_map:
        counts = co_map[source]
        m = max(counts.values())
        for t, c in counts.items():
            if t == source:
                continue
            sc = l_co * (c / m)
            if t not in best or sc > best[t][0]:
                best[t] = [sc, "co", 0]
    if "d" in modes and source in dep_map:
        counts = dep_map[source]
        m = max(counts.values())
        for t, c in counts.items():
            if t == source:
                continue
            sc = l_dep * (c / m)
            if t not in best or sc > best[t][0]:
                best[t] = [sc, "dep", 0]
    order = {"direct": 0, "transit": 1, "co": 2, "dep": 3}
    items = [(t, sc, origin, cnt) for t, (sc, origin, cnt) in best.items()]
    if strategy == "append":
        direct_items = sorted(
            (it for it in items if it[2] == "direct"),
            key=lambda it: (-it[1], -it[3], it[0]))
        extra_items = sorted(
            (it for it in items if it[2] != "direct"),
            key=lambda it: (-it[1], order[it[2]], -it[3], it[0]))
        return direct_items + extra_items
    items.sort(key=lambda it: (-it[1], order[it[2]], -it[3], it[0]))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="t,c,d")
    parser.add_argument("--strategy", choices=["append", "mix"], default="append")
    parser.add_argument("--l-transit", type=float, default=0.5)
    parser.add_argument("--l-co", type=float, default=0.3)
    parser.add_argument("--l-dep", type=float, default=0.3)
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
    co_map, dep_map = build_aux(train)
    print(f"co_map sources={len(co_map)}  dep_map sources={len(dep_map)}")

    def ranked_for(query, modes):
        source = query["rem_lib"]
        t0 = parse_time(query["time_stamp"])
        reason = (query.get("reason_label") or "").strip()
        return expand_rank(kb, co_map, dep_map, source, t0, reason, modes,
                           args.l_transit, args.l_co, args.l_dep, args.strategy)

    def eval_queries(queries, modes):
        cov = h1 = h3 = mrr = ndcg3 = ndcg10 = 0.0
        for q in queries:
            ranked = ranked_for(q, modes)
            pos = next((i for i, (t, _, _, _) in enumerate(ranked, 1)
                        if t == q["add_lib"]), None)
            if pos is None:
                continue
            cov += 1
            h1 += pos == 1
            h3 += pos <= 3
            mrr += 1 / pos
            if pos <= 3:
                ndcg3 += 1 / math.log2(pos + 1)
            if pos <= 10:
                ndcg10 += 1 / math.log2(pos + 1)
        n = len(queries)
        return cov, h1, h3, mrr, ndcg3, ndcg10, n

    rows_cfg = [("M0", "x"), ("M7t", "t"), ("M7c", "c"), ("M7tc", "t,c"),
                ("M7tcd", "t,c,d")]
    print(f"\n=== covered test set (true target in candidates) | tau 730 | "
          f"strategy={args.strategy} | l_t={args.l_transit} l_c={args.l_co} l_d={args.l_dep} ===")
    print(f"{'combo':<8}{'cov':>6}{'Hit@1':>8}{'Hit@3':>8}{'MRR':>8}"
          f"{'NDCG@3':>8}{'NDCG@10':>8}")
    for name, modes in rows_cfg:
        cov, h1, h3, mrr, ndcg3, ndcg10, _ = eval_queries(test, modes)
        print(f"{name:<8}{int(cov):>6}{h1/cov:>8.3f}{h3/cov:>8.3f}{mrr/cov:>8.4f}"
              f"{ndcg3/cov:>8.4f}{ndcg10/cov:>8.4f}")

    print(f"\n=== all-query test set (uncovered count as 0) ===")
    print(f"{'combo':<8}{'n':>6}{'Hit@1':>8}{'MRR':>8}{'NDCG@10':>8}")
    for name, modes in rows_cfg:
        cov, h1, h3, mrr, ndcg3, ndcg10, n = eval_queries(test, modes)
        print(f"{name:<8}{n:>6}{h1/n:>8.3f}{mrr/n:>8.4f}{ndcg10/n:>8.4f}")

    print(f"\n=== cold-start subset (no history edge for the source) ===")
    cold = [q for q in test if not kb.get(q["rem_lib"])]
    print(f"{'combo':<8}{'n':>6}{'recov':>8}{'Top10':>8}{'NDCG@10':>8}")
    for name, modes in rows_cfg:
        cov, h1, h3, mrr, ndcg3, ndcg10, n = eval_queries(cold, modes)
        rec = cov / n
        top10 = sum(1 for q in cold
                    if next((i for i, (t, _, _, _) in enumerate(ranked_for(q, modes), 1)
                             if t == q["add_lib"]), 0) <= 10)
        print(f"{name:<8}{n:>6}{rec:>8.3f}{top10/n:>8.3f}{ndcg10/n:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
