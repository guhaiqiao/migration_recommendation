#!/usr/bin/env bash
# 一键复现第四章全部实验：依次运行 scripts/ 下脚本，stdout 存入 outputs/。
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p outputs
cd scripts

run() {
    local name="$1"; shift
    echo "=================================================================="
    echo "==  $name"
    echo "=================================================================="
    python3 "$@" 2>&1 | tee "../outputs/${name}.log"
    echo
}

run reason_stratified      reason_stratified.py
run m8_reason_routing      m8_reason_routing.py
run m9_reason_spotlight    m9_reason_spotlight.py
run m10_signal_scenarios   m10_signal_scenarios.py
run m11_full_frequency     m11_full_frequency.py
run m12_per_language_full  m12_per_language_full.py
run per_language_ablation  per_language_ablation.py
run survival_analysis      survival_analysis.py --plot ../figures/km_survival.png
run n_significance_fa      n_significance_fa.py

echo "全部脚本执行完毕，完整输出在 outputs/*.log。"
echo "运行:  cd .. && python3 check_replication.py    # 与文档数值逐项校验"
