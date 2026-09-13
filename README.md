# 第四章实验复现包（Replication Package）

本项目复现迁移目标推荐的全部实验：数据集 + 推荐方法 + 实验结果。

## 目录结构

```
replication/
├── README.md                     # 本文件
├── run_all.sh                    # 一键运行全部实验脚本
├── check_replication.py          # 自动校验：脚本输出 vs 文档数值（71 项）
├── data/                         # 数据集快照（4 个语言正例 CSV + 非迁移替换对 + SHA256SUMS）
├── scripts/                      # 实验脚本副本（路径已改为包内相对路径）
├── expected/expected_metrics.json# 文档中记录的预期指标快照
├── figures/km_survival.png       # 生存分析 KM 曲线（可重绘）
└── outputs/                      # run_all.sh 的输出日志（运行后生成）
```

## 环境要求

- Python 3.9+（实验在 miniconda 的 Python 3.9 上完成）
- 标准库即可跑通全部排序实验；仅 `survival_analysis.py --plot` 绘图需要 `numpy` 与 `matplotlib`（3.9.4 验证过）

## 快速开始

```bash
cd replication
bash run_all.sh                    # 1. 运行全部脚本，输出到 outputs/*.log
python3 check_replication.py       # 2. 与文档数值逐项校验，应 71 PASS / 0 FAIL
```

校验器逐脚本解析 stdout 中的关键指标并与 `expected/expected_metrics.json` 比对（容差 0.001），任一不符即 FAIL 且退出码非 0。全套脚本运行时间 < 15 秒。

## 数据

四个语言生态的迁移标注数据集快照（合计 8,059，**仅保留 `is_migration=1` 的正例行**，按语言命名）：

| 文件 | 行数 | 时间范围 |
|---|---|---:|---|
| `python_positive.csv` | 3,827 | 1992-03 ~ 2021-07 |
| `java_positive.csv` | 1,420 | 2006-12 ~ 2021-07 |
| `js_positive.csv` | 641 | 2012-11 ~ 2021-07 |
| `c_positive.csv` | 2,171 | Unix 时间戳（`commit` 字段，全为正例） |

- **`non_migration_replacements.csv`**（4,381 行）：非迁移替换对（`dataset,time_stamp,rem_lib,add_lib`），仅用于 Fa 基线（含非迁移替换的全量频率）构建 KB_all；主 CSV 中已剔除这些行，不参与正例划分。
- 完整性校验：`sha256sum -c data/SHA256SUMS`
- 原始来源：costMeasure 项目数据目录（`/fast/guhaiqiao/data/dataset/`，符号链接自 `/home/guhaiqiao/costMeasure/data/`）。标注列（`is_migration`/`confidence`/`reason_label`）于 2026-09-07/08 追加。
- 字段：`repo_name`（C 语言数据为 `commit`）、`commit_sha`、`time_stamp`、`rem_lib`、`add_lib`、`commit_message`、`domain`、`is_migration`、`confidence`、`reason_label`（13 类原因之一或空）。

## 实验设置

- **分层时间切分**：每语言单独按 `time_stamp` 排序，训练 54% / 验证 11% / 测试 35%（训练 4,350 / 验证 884 / 测试 2,825；测试 covered 1,675，冷启动 917）；
- **反泄漏**：知识库仅由训练期迁移事件构建；
- **信号**：F = 频次 min-max；T = 最近事件衰减，τ=730 天；R = 原因标签精确匹配率；融合权重 w_R = 1/3；
- **基线**：Random（候选随机打乱，seed 42）、Fa（含非迁移替换的全量频率，无迁移过滤）；
- **FTR\***：R 按原因分类（验证集逐层比较 FTR vs FT 的 MRR 决定开关；空原因层常闭）；
- **口径**：covered（真值在候选中）为主口径，同时报告 all-query 与宏平均（13 类原因等权）。

## 脚本 ↔ 复现指标

| 脚本 | 复现指标 |
|---|---|
| `reason_stratified.py`  | 各原因层 ΔMRR、ALL 行 |
| `m8_reason_routing.py`  | full/annotated/empty 三口径 routed 指标 |
| `m9_reason_spotlight.py`  | tiebreak 44 例（ε=0.05）、宏平均、修复/破坏计数；`--eps 0.02/0.10` 复现 ε 曲线 |
| `m10_signal_scenarios.py`  | direction_shift / f_flat / reason_active |
| `m11_full_frequency.py`  | 10 变体 × 4 指标 × 4 场景（Random + Fa 基线） |
| `m12_per_language_full.py` | 四语言 + pooled 的 10 变体 × 4 指标 |
| `per_language_ablation.py` | 分语言三口径（full/annotated/tiebreak）Hit@1/MRR |
| `survival_analysis.py`  | 8,059 受试者 log-rank p 值；`--plot` 重绘 `figures/km_survival.png` |
| `n_significance_fa.py`  | Fa vs F 的配对 Bootstrap 95% CI 与 McNemar p |

