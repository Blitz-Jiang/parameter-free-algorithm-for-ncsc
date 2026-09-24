# UTR5-fixed：完整 mpmath 固定次数实现

2026-09-24，按用户要求保留论文的固定 N_tr 次成功减半，使用高精度避免此前 float64 平台。

## 名称与范围

- `utr5-fixed`、`utr5-fix`：同一 mpmath 实现，默认80位十进制有效数字。
- `utr5-fixed-float64`：保留的 float64 严格次数诊断版。
- `utr5`：原先 float64 冻结总目标版，未替换。
- `utr5-early`：原先 AR 内部早停版，未替换。

新增算法 `algorithms/utr5_fixed_mp.py`，普通 SCAR 后端 `inner_solvers/scar_mp.py`，
信赖域后端 `subproblem_solvers/trs_mp.py`，问题后端 `problems/coslogcosh_mp.py`。

目前只为本仓库 canonical CosLogCosh 目标提供解析高精度后端。
算法通过显式 oracle_factory 注入问题，其他目标必须提供相应后端，不会自动套用此目标公式。
MP 后端不调用 problem.ell/mu/rho 等正则性界作算法参数；payoff 本身的 mu_y 等系数属于目标定义。

## 真正保留高精度的部分

x、y、SCAR 的 nu/M、动量、残差、目标值、梯度、Hessian、TR 特征分解与乘子搜索，
全部在独立 mpmath context 中运算。中间步骤不转回 float64。
每个工作候选完整执行 N_tr 次成功 SCARHalf；没有总残差提前结束，也没有 AR early stopping。
每次 AR 重启动量，跨调用仅保留曲率校准；拒绝候选后从旧内层点刷新，保留校准。
稳定函数余项采用精度相关级数，避免直接函数值相减造成另一类消去误差。

仅输入的既有 torch 系数/初值转为高精度，以及最终 AlgorithmResult 为兼容评价器转回 torch。
输入所代表的浮点系数被原样提升精度，不重新随机生成问题。history 另存完整 x_mp/y_mp 字符串。
最终报告的公共精度来自既有独立 float64 evaluator，评价的是最终舍入的输出点。

80位仍是有限精度，不声称对任意实例都足够。可用 --mp-dps 提高，最低40位。
保留曲率尝试、AR 子问题和外层预算；失败明确返回未收敛，不把残差平台算作减半成功。
TR 用数值 KKT 验证及精度相关容差；不宣称精确算术实现。

## 计数

计数器在高精度问题 grad_y 函数入口累加，包含初始、候选、secant、普通 AR、拒绝刷新及验证查询。
每一阶段又独立记录差值，核对总和；不靠填入浮点 problem 的计数器模拟查询。
实际 function 查询也计数。修正 oracle 复用已有 grad_y，不虚增内层梯度次数。
每次向量梯度查询计为1次，沿用其他算法的 oracle 定义。

输出明确标注 arithmetic=mpmath、decimal_digits、oracle_backend。
高精度解析后端和其他算法的 float64/autograd 后端不同；oracle 次数可按相同定义报告，
单次 wall time 不能解释为算法本身的公平速度排名。

## 运行

依赖 torch、mpmath；本次使用环境 `/Users/koa/miniforge3/envs/utr-scar/bin/python`。

```sh
python numerical_experiment/scripts/run_utr5_fixed.py --d 5 --epsilon 0.001 --mp-dps 80
python numerical_experiment/scripts/run_comparision.py --algorithms utr5-fix --d 10 --mu-y 0.1 --epsilon 0.001 --max-iterations 100
python numerical_experiment/src_test/test_utr5_fixed_mp.py
```

## 实际验证

四项性质测试通过：解析梯度与高精度数值微分/torch oracle 对照，极小步函数余项，
TR 正定/不定 hard case/奇异情形，严格 SCAR 产生小于1e-20残差，
完整迭代、固定次数合同、独立查询计数器对账以及全局精度 context 不被修改。

| 问题（seed=2，epsilon=0.001） | 外层子问题 | 内层梯度 | 成功减半调用 | 公共精度 |
|---|---:|---:|---:|---|
| d=5，mu_y=1（此前 float64 卡住） | 6 | 8684 | 94 | 通过 |
| d=10，mu_y=0.1 | 10 | 11230 | 160 | 通过 |

第一例参考梯度约3.6543e-6、Hessian 最小特征值约0.94478；
第二例参考梯度约3.6267e-6、Hessian 最小特征值约0.21334。
记录：`/tmp/utr5_fixed_mp_d5.json`、`/tmp/utr5_fixed_mp_d10.json`。
本次完整高精度实现取代“只有冻结内层重放”的旧状态，不代表其他问题类型已有高精度支持。
