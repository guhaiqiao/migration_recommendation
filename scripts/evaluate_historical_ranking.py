#!/usr/bin/env python3
"""Evaluate the historical-evidence ranking (M0) on labeled migration instances.

Protocol follows experiments.md: strict time split of python_mig_history.csv
(5,805 instances) into train 3,150 / val 646 / test 2,009. Training positives
build the historical source->target edge knowledge base; test positives are
queries. Candidates of a query are ranked by S_H = (F + T + R) / 3, where

  F  frequency      min-max normalised count of the historical source->target edge
  T  recency        min-max normalised time decay exp(-delta_days / tau) of the edge's latest occurrence
  R  reason match   fraction of the edge's instances whose reason_label equals the query reason

When the query has no reason label, S_H = (F + T) / 2.
Metrics: Hit@1, Hit@3, MRR, NDCG@3 (covered queries and all queries).
"""
import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from classify_reasons import classify_message

# classify_reasons 类别 -> 标注集 taxonomy 标签名
REASON_MAP = {
    "deprecation": "Deprecation",
    "bug or issue": "BugIssue",
    "security": "Security",
    "functionality": "Functionality",
    "usability": "Usability",
    "performance": "Performance",
    "activity": "Activity",
    "popularity": "Popularity",
    "size": "SizeComplexity",
    "integration": "Integration",
    "simplification": "Simplification",
    "organization": "Organization",
    "license": "License",
}


def parse_time(value: str) -> dt.datetime:
    value = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        parsed = None
    if parsed is None:
        # Unix 时间戳形式，如 "1584105712+0800"
        match = re.match(r"^(\d{9,11})([+-]\d{4})?$", value)
        if match:
            seconds = int(match.group(1))
            tz = dt.timezone.utc
            if match.group(2):
                sign = 1 if match.group(2)[0] == "+" else -1
                hours, minutes = int(match.group(2)[1:3]), int(match.group(2)[3:5])
                tz = dt.timezone(sign * dt.timedelta(hours=hours, minutes=minutes))
            parsed = dt.datetime.fromtimestamp(seconds, tz)
        else:
            parsed = dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)  # 统一为带时区，避免混用
    return parsed


def load_instances(path: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def load_mined(path: Path) -> list[dict]:
    """Load confidence=high rows from a *_migrations_dedup.csv into instance form."""
    csv.field_size_limit(sys.maxsize)
    rows = []
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row.get("confidence") != "high":
                continue
            matched = classify_message(row.get("commit_message", ""))
            reason = REASON_MAP.get(matched[0], "") if matched else ""
            rows.append({
                "rem_lib": row["removed_dependency"],
                "add_lib": row["added_dependency"],
                "time_stamp": row["committed_at"],
                "commit_message": row.get("commit_message", ""),
                "is_migration": "1",
                "reason_label": reason,
                "_dataset": path.stem,
            })
    return rows


def language_of(stem: str) -> str:
    lowered = stem.lower()
    if lowered.startswith("python"):
        return "python"
    if lowered.startswith(("js", "javascript")):
        return "js"
    if lowered.startswith("java"):
        return "java"
    if lowered.startswith("c_"):
        return "c"
    return lowered


def positive(instances: list[dict]) -> list[dict]:
    return [row for row in instances if row.get("is_migration") in ("1", "1.0", "True", "true")]


def build_knowledge_base(rows: list[dict]) -> dict:
    """source -> {target: {"count": n, "times": [...], "reasons": Counter}}."""
    kb = defaultdict(lambda: defaultdict(lambda: {"count": 0, "times": [], "reasons": defaultdict(int)}))
    for row in rows:
        source, target = row["rem_lib"], row["add_lib"]
        if not source or not target:
            continue
        edge = kb[source][target]
        edge["count"] += 1
        edge["times"].append(parse_time(row["time_stamp"]))
        reason = (row.get("reason_label") or "").strip()
        edge["reasons"][reason] += 1
    return kb


def minmax_normalise(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.0] * len(values)
    return [(v - low) / (high - low) for v in values]


def rank_candidates(kb: dict, source: str, t0: dt.datetime, reason: str, tau: int,
                   w_reason: float = 1 / 3, epsilon: float = None,
                   signals: str = "FTR") -> list[str]:
    """Rank candidates by the selected signals (subset of F/T/R).

    signals == "FTR" keeps the original weighted formula
    S = w_R*R + (1-w_R)*(F+T)/2; other subsets average the selected signals.
    With epsilon set (FTR only), candidates whose |S diff| < epsilon are
    reordered by reason match R.
    """
    edges = kb.get(source)
    if not edges:
        return []
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
    comps = {"F": f_norm, "T": t_norm, "R": r_scores}
    selected = [c for c in signals.upper() if c in comps]
    items = []
    for idx, target in enumerate(targets):
        if signals.upper() == "FTR" and reason:
            base = (f_norm[idx] + t_norm[idx]) / 2
            score = (1 - w_reason) * base + w_reason * r_scores[idx]
        elif selected:
            score = sum(comps[c][idx] for c in selected) / len(selected)
        else:
            score = 0.0
        items.append({"target": target, "S": score, "R": r_scores[idx],
                      "count": edges[target]["count"]})
    items.sort(key=lambda item: (-item["S"], -item["count"], item["target"]))
    if epsilon is not None and reason and signals.upper() == "FTR":
        # 带内重排：与组内最高分差 < epsilon 的归为一组，组内按 R 降序
        bands = []
        for item in items:
            if bands and abs(bands[-1][0]["S"] - item["S"]) < epsilon:
                bands[-1].append(item)
            else:
                bands.append([item])
        ranked = []
        for band in bands:
            band.sort(key=lambda item: (-item["R"], -item["count"], item["target"]))
            ranked.extend(item["target"] for item in band)
        return ranked
    return [item["target"] for item in items]


def evaluate(queries: list[dict], kb: dict, tau: int, w_reason: float = 1 / 3,
             epsilon: float = None, signals: str = "FTR") -> dict:
    hits1 = hits3 = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    covered = 0
    total = len(queries)
    for query in queries:
        source = query["rem_lib"]
        true_target = query["add_lib"]
        t0 = parse_time(query["time_stamp"])
        reason = (query.get("reason_label") or "").strip()
        ranked = rank_candidates(kb, source, t0, reason, tau, w_reason, epsilon, signals)
        if not ranked:
            continue
        covered += 1
        rank = None
        for pos, target in enumerate(ranked, start=1):
            if target == true_target:
                rank = pos
                break
        if rank is None:
            continue
        if rank == 1:
            hits1 += 1
        if rank <= 3:
            hits3 += 1
            ndcg_sum += 1.0 / math.log2(rank + 1)
        mrr_sum += 1.0 / rank
    return {
        "queries": total,
        "covered": covered,
        "uncovered": total - covered,
        "hit@1": hits1 / total,
        "hit@3": hits3 / total,
        "mrr": mrr_sum / total,
        "ndcg@3": ndcg_sum / total,
        "hit@1_covered": hits1 / covered if covered else 0.0,
        "hit@3_covered": hits3 / covered if covered else 0.0,
        "mrr_covered": mrr_sum / covered if covered else 0.0,
        "ndcg@3_covered": ndcg_sum / covered if covered else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, action="append", required=True,
                        help="Labeled instance CSV; repeatable for multiple languages")
    parser.add_argument("--mined", type=Path, action="append", default=[],
                        help="data/xxx_migrations_dedup.csv mined by this pipeline; "
                             "only confidence=high rows are used")
    parser.add_argument("--train-frac", type=float, default=0.54)
    parser.add_argument("--val-frac", type=float, default=0.11)
    parser.add_argument("--tau", type=int, default=730, help="Recency decay half-life in days")
    parser.add_argument("--w-reason", type=float, default=1 / 3,
                        help="Weight of the reason-match signal R; F and T share the rest")
    parser.add_argument("--tiebreak-epsilon", type=float, default=None,
                        help="In-band re-ranking: reorder by R within candidates whose S_H diff < epsilon")
    parser.add_argument("--reason-only", action="store_true",
                        help="Evaluate only queries whose commit message carries a reason label")
    parser.add_argument("--covered-only", action="store_true",
                        help="Evaluate only queries whose source has historical edges in the training KB")
    parser.add_argument("--signals", type=str, default="FTR",
                        help="Which signals to use, subset of F/T/R (e.g. F, T, R, FT, FR, TR, FTR)")
    args = parser.parse_args()

    instances = []
    for path in args.instances:
        for row in load_instances(path):
            row["_dataset"] = path.stem
            instances.append(row)
    for path in args.mined:
        instances.extend(load_mined(path))
    # 按 is_migration=1 过滤
    instances = positive(instances)
    # 按语言分层：各语言按时间排序后独立切分
    by_language = defaultdict(list)
    for row in instances:
        by_language[language_of(row["_dataset"])].append(row)
    train_rows, val_rows, test_rows = [], [], []
    for language, rows in by_language.items():
        rows.sort(key=lambda row: parse_time(row["time_stamp"]))
        train_n = int(len(rows) * args.train_frac)
        val_n = int(len(rows) * args.val_frac)
        train_rows.extend(rows[:train_n])
        val_rows.extend(rows[train_n:train_n + val_n])
        test_rows.extend(rows[train_n + val_n:])
        print(f"split/{language}: {len(rows)} -> train {train_n} / val {val_n} / test {len(rows) - train_n - val_n}")
    kb = build_knowledge_base(train_rows)

    if args.reason_only:
        val_rows = [row for row in val_rows if (row.get("reason_label") or "").strip()]
        test_rows = [row for row in test_rows if (row.get("reason_label") or "").strip()]
    if args.covered_only:
        val_rows = [row for row in val_rows if kb.get(row["rem_lib"])]
        test_rows = [row for row in test_rows if kb.get(row["rem_lib"])]

    print(f"instances {len(instances)} | train {len(train_rows)} | val {len(val_rows)} | "
          f"test {len(test_rows)} | tau {args.tau} | w_reason {args.w_reason} | eps {args.tiebreak_epsilon} "
          f"| signals {args.signals}")
    val_metrics = evaluate(val_rows, kb, args.tau, args.w_reason, args.tiebreak_epsilon, args.signals)
    test_metrics = evaluate(test_rows, kb, args.tau, args.w_reason, args.tiebreak_epsilon, args.signals)
    print("validation:", json.dumps(val_metrics, ensure_ascii=False))
    print("test:      ", json.dumps(test_metrics, ensure_ascii=False))
    test_by_language = defaultdict(list)
    for row in test_rows:
        test_by_language[language_of(row["_dataset"])].append(row)
    for language, subset in sorted(test_by_language.items()):
        print(f"test/{language}: ", json.dumps(evaluate(subset, kb, args.tau, args.w_reason, args.tiebreak_epsilon, args.signals), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
