#!/usr/bin/env python3
"""复现校验器（replication checker）。

依次运行 scripts/ 下的全部实验脚本，解析其 stdout 中的关键指标，
与 expected/expected_metrics.json 中记录的文档数值逐项比对（容差 1e-3）。

用法:
    python3 check_replication.py            # 全部校验
    python3 check_replication.py --skip-plots   # 生存分析不重绘图
输出 PASS/FAIL 汇总表，任何一项失败则退出码为 1。
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
EXPECTED_PATH = ROOT / "expected" / "expected_metrics.json"
TOL = 1e-3


def run(script: str, extra_args=()) -> str:
    proc = subprocess.run(
        [sys.executable, script, *extra_args],
        cwd=SCRIPTS, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{script} 退出码 {proc.returncode}:\n{proc.stderr}")
    return proc.stdout


# ------------------------- stdout 解析器 -------------------------

def p_stratified(out: str) -> dict:
    """reason_stratified.py: 原因分层诊断表（dMRR 为第 9 列）。"""
    res = {}
    reasons = {"Functionality", "Simplification", "BugIssue", "Security",
               "Deprecation", "Usability", "Performance"}
    for line in out.splitlines():
        f = line.split()
        if len(f) == 9 and f[0] == "ALL":
            res["ALL.Hit@1"] = float(f[3])
            res["ALL.MRR"] = float(f[5])
        elif len(f) == 9 and f[0] in reasons:
            res[f"{f[0]}.dMRR"] = float(f[8])
    return res


def _view_table(out: str, marker: str) -> dict:
    res = {}
    start = out.find(marker)
    if start < 0:
        return res
    for line in out[start:].splitlines():
        f = line.split()
        if len(f) == 7 and f[0] in {"full", "annotated", "empty",
                                    "r-present", "r-nontrivial", "tiebreak",
                                    "micro"} and f[1] in {"FT", "FTR", "routed"}:
            res[f"{f[0]}.{f[1]}.Hit@1"] = float(f[3])
            res[f"{f[0]}.{f[1]}.MRR"] = float(f[5])
    return res


def p_m8(out: str) -> dict:
    return _view_table(out, "=== test results (covered) ===")


def p_m9(out: str) -> dict:
    res = _view_table(out, "=== M9 views")
    for line in out.splitlines():
        m = re.search(r"^(FT|FTR|routed)\s+strata=\s*\d+\s+macro-MRR=([\d.]+)",
                      line)
        if m:
            res[f"macro.{m.group(1)}"] = float(m.group(2))
        m2 = re.search(r"fixed (\d+) broken (\d+)", line)
        if m2:
            res["fixed"] = float(m2.group(1))
            res["broken"] = float(m2.group(2))
    return res


def _cell_table(cols, scenarios, nseg):
    """通用单元格表格解析：行首为场景名，其后每列 a/b[/c[/d]]。"""
    def p(out: str) -> dict:
        res = {}
        for line in out.splitlines():
            f = line.split()
            if len(f) < 2 + len(cols) or f[0] not in scenarios:
                continue
            for name, cell in zip(cols, f[2:2 + len(cols)]):
                parts = cell.split("/")
                if len(parts) != nseg:
                    continue
                res[f"{f[0]}.{name}"] = [float(x) for x in parts]
                res[f"{f[0]}.{name}.Hit@1"] = float(parts[0])
                res[f"{f[0]}.{name}.MRR"] = float(parts[2])
        return res
    return p


p_m10 = _cell_table(["Rnd", "F", "T", "R", "FT", "FTR", "FTR*"],
                    {"full", "direction_shift", "f_flat", "reason_active",
                     "f_flat_reason"}, 3)
p_m11 = _cell_table(["Rnd", "Fa", "F", "T", "R", "FT", "FR", "TR", "FTR",
                     "FTR*"],
                    {"full", "direction_shift", "f_flat", "reason_active"}, 4)
p_m12 = _cell_table(["Rnd", "Fa", "F", "T", "R", "FT", "FR", "TR", "FTR",
                     "FTR*"],
                    {"python", "java", "js", "c", "pooled"}, 4)


def p_pla(out: str) -> dict:
    """per_language_ablation.py: 行 = lang scope n 9×(Hit@1/MRR)。"""
    res = {}
    cols = ["Rnd", "F", "T", "R", "FT", "FR", "TR", "FTR", "FTR*"]
    for line in out.splitlines():
        f = line.split()
        if len(f) != 12 or f[1] not in {"full", "annotated", "tiebreak"}:
            continue
        lang, scope = f[0], f[1]
        for name, cell in zip(cols, f[3:]):
            h1, mrr = cell.split("/")
            res[f"{lang}.{scope}.{name}.Hit@1"] = float(h1)
            res[f"{lang}.{scope}.{name}.MRR"] = float(mrr)
    return res


def p_survival(out: str) -> dict:
    res = {}
    m = re.search(r"subjects = (\d+)", out)
    if m:
        res["subjects"] = float(m.group(1))
    for block in re.split(r"(?m)^=== ", out):
        m = re.match(r"(F_top|T_top|R_sup)", block)
        if not m:
            continue
        key = m.group(1)
        pv = re.search(r"p = ([\d.]+)", block)
        if pv:
            res[f"{key}.p"] = float(pv.group(1))
        t730 = re.search(r"at 730d: ([\d.]+) vs ([\d.]+)", block)
        if t730:
            res[f"{key}.730d.1"] = float(t730.group(1))
            res[f"{key}.730d.0"] = float(t730.group(2))
    return res


def p_sigfa(out: str) -> dict:
    res = {}
    for line in out.splitlines():
        m = re.search(r"^(\w+): n=\d+.*MRR delta ([+-]?\d+\.\d+) "
                      r"95% CI \[([+-]?\d+\.\d+),([+-]?\d+\.\d+)\]", line)
        if m:
            res[f"{m.group(1)}.mrr_delta"] = float(m.group(2))
            res[f"{m.group(1)}.ci_lo"] = float(m.group(3))
            res[f"{m.group(1)}.ci_hi"] = float(m.group(4))
        m2 = re.search(r"^(\w+): .*p=([\d.]+)\s*$", line)
        if m2:
            res[f"{m2.group(1)}.p"] = float(m2.group(2))
    return res


RUNS = [
    ("reason_stratified.py", (), p_stratified),
    ("m8_reason_routing.py", (), p_m8),
    ("m9_reason_spotlight.py", (), p_m9),
    ("m9_reason_spotlight.py", ("--eps", "0.02"), p_m9),
    ("m9_reason_spotlight.py", ("--eps", "0.10"), p_m9),
    ("m10_signal_scenarios.py", (), p_m10),
    ("m11_full_frequency.py", (), p_m11),
    ("m12_per_language_full.py", (), p_m12),
    ("per_language_ablation.py", (), p_pla),
    ("survival_analysis.py", (), p_survival),
    ("n_significance_fa.py", (), p_sigfa),
]


def matches(expect, actual) -> bool:
    if isinstance(expect, list):
        return (isinstance(actual, list) and len(expect) == len(actual)
                and all(abs(e - a) <= TOL for e, a in zip(expect, actual)))
    return isinstance(actual, (int, float)) and abs(expect - actual) <= TOL


def fmt(v) -> str:
    if isinstance(v, list):
        return "/".join(f"{x:.4f}" for x in v)
    return f"{v:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-plots", action="store_true",
                    help="生存分析不重绘 KM 曲线")
    args = ap.parse_args()

    expected = json.loads(EXPECTED_PATH.read_text())
    n_pass = n_fail = 0

    for script, extra, parser in RUNS:
        key = script if not extra else f"{script}--eps{extra[1]}"
        if key not in expected:
            continue
        print(f"\n== {key} ==")
        try:
            actual = parser(run(script, extra))
        except RuntimeError as e:
            print(f"  !! 运行失败: {e}")
            n_fail += len(expected[key])
            continue
        for check in expected[key]:
            desc, expect = check["desc"], check["expect"]
            got = actual.get(desc)
            ok = got is not None and matches(expect, got)
            status = "PASS" if ok else "FAIL"
            if ok:
                n_pass += 1
            else:
                n_fail += 1
            print(f"  [{status}] {desc:35s} expect {fmt(expect):>28s}  "
                  f"got {fmt(got) if got is not None else '<missing>'}")

    print(f"\n===== 汇总: {n_pass} PASS / {n_fail} FAIL =====")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
