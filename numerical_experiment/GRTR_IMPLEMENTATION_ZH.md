# GRTR 实现说明（2026-09-24）

本地来源：`related_papers/GRTR.pdf`，33页，SHA256
`0fd72b53dd0a391335d0dda0d5a3fa3174084fbb8a46161db89df082d4953522`。
实现 Algorithm 1 的 GRTR，理论参数采用 Lemma 2.5；不是 IGRTR 或 LMNegCur。

## 原文与代码对应

源码 `algorithms/grtr.py`，直接继承 NCSCAlgorithm，注入 NesterovAGD 和 TRSubproblemSolver。

| 原文 | 实现 |
|---|---|
| Lemmas 2.1–2.2 | kappa=ell/mu，L1=(1+kappa)*ell，LH=rho*(1+kappa)^2，L2=rho*(1+kappa)^3 |
| (2.11) | sigma=sqrt(L2)/2，radius_scale=1/(4*sqrt(L2)) |
| (2.11) | epsilon_g=min(1/96,sqrt(L2)/(16*L1))*epsilon^(3/2)，epsilon_H=sqrt(L2*epsilon)/12 |
| Lemma 2.4 | A=min(epsilon_g/ell,epsilon_H/(2*LH))，固定步数 AGD，warm start y，重启动量 |
| Algorithm 1 Step 3 | g=grad_x f(x,y)，H=Hxx-Hxy solve(Hyy,Hyx)，非修正梯度 |
| (2.3) | B=H+sigma*sqrt(norm(g))*I，radius=radius_scale*sqrt(max(norm(g),epsilon)) |
| Lemma 2.6 | norm(g)<=epsilon 且 lambda<=sqrt(L2*epsilon) 时返回当前 x_t |
| 非终止步 | x=x+s，无函数值接受测试，无自适应 sigma |

返回当前点的理论界为 norm(grad P)<=97*epsilon/96，
lambda_min(Hessian P)>=-19*sqrt(L2*epsilon)/12；不能把原生 converged 视为公共精度已通过。

## 明确的实现选择及数值边界

- 限定 0<epsilon<=1，使上述 epsilon^(3/2) 误差预算可用于论文小精度结论。
- 首次距离上界使用 norm(grad_y f(x0,y0))/mu，额外一次查询计入成本。
- 固定步数使用原文 sqrt(kappa+1) 前因子，向上取整并截断到非负，允许0步。后续距离上界用 A+kappa*norm(s_prev)。
  不使用残差早停；不查询终点梯度，每次 N 步恰计 N 次查询，residual=None。
- TR 复用现有 dense 求解器，并额外检查站立性、半径可行性、互补性、
  模型加乘子后的半正定性和乘子非负性。默认归一化 KKT 容差 1e-8；
  半径检查使用相对误差。检查失败明确返回未收敛，不静默接受坏解。
- TR 为数值解而非精确算术全局解；实现不因此宣称有额外的有限精度复杂度定理。
- `max_iterations` 是 TR 调用预算，含终止测试所需的一次求解；
  `n_updates` 另记实际移动次数。预算不足、浮点停滞与求解失败均有 termination_reason。
- 正常终止时 y 与返回的当前 x 匹配；预算耗尽发生在更新后时 y 可能属于前一 x，
  history 的 y_matches_output_x 明确记录。公共评价在返回 x 重新求内层。
- 包装器统计实际 grad_y 调用，并与内层返回计数对账；包含初始化和终端残差查询。

## 运行

在项目根目录及已安装 torch 的环境中：

```sh
python numerical_experiment/scripts/run_grtr.py --d 2 --epsilon 0.001 --max-iterations 2000
python numerical_experiment/scripts/run_comparision.py --algorithms utr5 hsda grtr --d 2 --epsilon 0.001 --max-iterations 2000
python numerical_experiment/src_test/test_grtr.py
```

可用 `--grtr-max-inner-steps` 设置每次内层的固定步数预算；若论文所需步数超预算，
不会截断后继续运行。原来的默认参比名单保持兼容，请显式选择本轮参比算法。

## 已执行验收

五项测试：终止返回当前点且计数一致；零梯度负曲率点的边界步；
多步收敛和已知 y* 下的初始/warm-start 精度；坏子问题结果及预算失败；
零步正常终止与 kappa=1。

二维 CosLogCosh，seed=2，epsilon=0.001，float64：

| 方法 | 外层子问题数 | 内层梯度数 | 参考梯度范数 | 参考 Hessian 最小特征值 |
|---|---:|---:|---:|---:|
| UTR5 | 6 | 3050 | 1.261e-6 | 1.02654 |
| HSDA | 485 | 10196 | 8.205e-4 | 1.02647 |
| GRTR | 82 | 3433 | 5.564e-4 | 1.02649 |

三者实际梯度查询与自报计数一致，均通过公共误差界感知的二阶精度评价。
原始本次输出为 `/tmp/grtr_hsda_utr5_smoke.json`。这是实现验收小例，不是正式性能结论。
正式加入 MCN 比较前，仍需统一现有 runner 中排除的 MCN 初始化成本口径。

MINIMAX-TRACE 仍暂缓，原函数值 oracle 和步长相关内层精度问题保留在五算法设计文档。
