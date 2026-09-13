#!/usr/bin/env python3
"""Classify high-confidence migration instances by the reason mentioned in their commit message.

Categories follow the migration-motivation taxonomy: deprecation, bug or issue,
security, functionality, usability, performance, activity, popularity, size,
integration, simplification, organization, license. A commit message may match
several categories (multi-label).
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

CATEGORIES = [
    ("deprecation", [r"deprecat", r"obsolete", r"\beol\b", r"end[- ]of[- ]life", r"\blegacy\b", r"no longer (?:supported|maintained)"]),
    ("bug or issue", [r"\bbugs?\b", r"bug[- ]?fix", r"\bfix(?:es|ed)?:?\s+#?\d+", r"\bresolves?:?\s+#?\d+",
                      r"\bissues?\b", r"\bcrash(?:es|ing)?\b", r"\berrors?\b", r"\bbroken\b",
                      r"\bworkaround\b", r"\bdefects?\b", r"\bregression"]),
    ("security", [r"secur", r"vulnerabilit", r"\bcve[- ]", r"exploit", r"\bxss\b", r"unsafe", r"\bsecure\b"]),
    ("functionality", [r"\bfeatures?\b", r"functionality", r"capabilit", r"missing (?:functionality|features?)"]),
    ("usability", [r"usab", r"user[- ]friendly", r"intuitive", r"\bux\b", r"ergonomic", r"convenien"]),
    ("performance", [r"performance", r"\bfaster\b", r"\bspeed\b", r"\bslow(?:er)?\b", r"latency", r"throughput", r"optimiz", r"benchmark", r"memory (?:usage|consumption)"]),
    ("activity", [r"\bactive(?:ly)?\b", r"activity", r"\bmaintain(?:ed|er|ers)?\b", r"maintenance", r"unmaintained", r"\bstale\b", r"\babandon(?:ed)?\b", r"inactive", r"\bdead (?:project|repo)\b", r"no (?:longer )?(?:commits|updates?)"]),
    ("popularity", [r"popular", r"widely (?:used|adopted)", r"\bcommunity\b", r"\bstars?\b", r"adoption", r"\badopt(?:ed|ing)?\b"]),
    ("size", [r"\bsize\b", r"\bsmaller\b", r"light[- ]?weight", r"\bbloat\b", r"\bheavy\b", r"\bfootprint\b", r"bundle size"]),
    ("integration", [r"integrat", r"compatib", r"interoperab", r"works? with", r"ecosystem", r"plugin (?:system|ecosystem)"]),
    ("simplification", [r"simplif", r"\bsimpler\b", r"reduce complexity", r"clean[- ]?up", r"consolidat"]),
    ("organization", [r"organi[sz]", r"reorgani", r"\brenam(?:e|ed|ing)\b", r"\bmoved\b", r"new home", r"ownership", r"maintainership", r"transferred"]),
    ("license", [r"licen[cs]e", r"\bmit\b", r"apache", r"\bgpl\b", r"\blgpl\b", r"\bbsd\b", r"proprietary", r"\blegal\b"]),
]


def classify_message(message: str) -> list[str]:
    """Return the categories mentioned in the message."""
    return [category for category, patterns in CATEGORIES
            if any(re.search(pattern, message, re.IGNORECASE) for pattern in patterns)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Migrations CSV (dedup output)")
    parser.add_argument("--output-rows", type=Path, help="Optional CSV with per-row matched categories")
    args = parser.parse_args()
    csv.field_size_limit(sys.maxsize)
    with args.input.open(encoding="utf-8", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row.get("confidence") == "high"]
    labels = []
    for row in rows:
        matched = classify_message(row.get("commit_message", ""))
        labels.append((row, matched))
    counts = Counter()
    for _, matched in labels:
        counts.update(matched)
    total = len(rows)
    none = sum(1 for _, matched in labels if not matched)
    print(f"high 实例共 {total} 条，未提及任何原因的 {none} 条（{100 * none / total:.1f}%）" if total else "no high rows")
    for category, count in counts.most_common():
        print(f"  {count:5d} ({100 * count / total:5.1f}%)  {category}" if total else category)
    if args.output_rows:
        args.output_rows.parent.mkdir(parents=True, exist_ok=True)
        with args.output_rows.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0].keys()) + ["reasons"])
            writer.writeheader()
            for row, matched in labels:
                writer.writerow({**row, "reasons": "|".join(matched)})
        print(f"per-row categories written to {args.output_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
