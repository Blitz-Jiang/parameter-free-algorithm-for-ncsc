# CosLogCosh 数值实验

仅保留 CosLogCosh 问题及各算法实现。旧数据、实验输出、其他问题和旧批量实验框架已移除。

## 运行

依赖：Python 3.10+、PyTorch、mpmath。以下从仓库根目录运行；`python` 应指向装有这些依赖的环境。

```bash
python numerical_experiment/scripts/run_comparision.py \
  --algorithms utr5-fixed utr5 utr5-early grtr hsda mcn \
  --d 10 --mu-y 0.1 --omega 1 --beta 0.5 --seed 2 \
  --epsilon 0.001 --max-iterations 30000 --threads 1 --mp-dps 80
```

输出默认写入被 Git 忽略的 `outputs/`。可用 `--output` 指定 JSON 路径。
单算法入口：`run_mcn.py`、`run_grtr.py`、`run_hsda.py`、`run_utr5.py`、`run_utr5_fixed.py`、`run_utr2.py`。
其他版本可在比较入口通过 `--algorithms` 选择。

## 目录

- `problems/CosLogCosh.py`：float64 问题；`coslogcosh_mp.py`：同一问题的高精度后端；`NCSC.py`：公共接口。
- `algorithms/`：保留 MCN、GRTR、HSDA、UTR2–5 和 UTR5-fixed；UTR5 依赖 UTR4/UTR3。
- `inner_solvers/`、`subproblem_solvers/`：AGD、SCAR 及 TR/cubic/特征子问题。
- `src_test/`：算法与数值正确性测试；其中的小型解析模型仅为测试夹具。
- `related_papers/`：参考论文，MINIMAX-TRACE 仍未实现。
- [完整伪代码](pseudocode/ALL_SIX_ALGORITHMS.md)：六个主要比较版本。

## 计数与精度

MCN、GRTR、HSDA 默认使用初始 residual/mu 距离上界，内层成本为 `1+sum(K_t)`。
固定步数 AGD 不查询终点 residual，日志值为 `None`。公共终点评价成本另计。
UTR5-fixed 使用 mpmath 80位，其余主要比较版本使用 float64。
各算法采用原生停止条件，公共精度通过情况由独立评价器报告；内部 epsilon 相同不代表理论常数相同。

## 核心回归测试

```bash
python numerical_experiment/src_test/test_cubic.py
python numerical_experiment/src_test/test_baseline_accounting.py
python numerical_experiment/src_test/test_grtr.py
python numerical_experiment/src_test/test_hsda.py
python numerical_experiment/src_test/test_utr5_fixed.py
python numerical_experiment/src_test/test_utr5_fixed_mp.py
```
