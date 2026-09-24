# HSDA 实现与验证

来源：`related_papers/HSDA.pdf`，26页，SHA256
`3afa87e3843ad414cad689d39e96467e236566c4524d3cd69c9337fc994809fe`。
实现 Algorithm 1 的 dense HSDA，不是 Algorithm 2 的 IHSDA。

## 公式对应

| 原文 | 实现 |
|---|---|
| Lemma 2.1 | L1=(1+kappa)*ell，LH=rho*(1+kappa)^2，L2=rho*(1+kappa)^3 |
| Theorem 2.1 | alpha=sqrt(L2*epsilon)，Lambda=sqrt(epsilon/L2)，epsilon_g=epsilon/12，epsilon_H=alpha/12 |
| Lemma 2.3 | A=min(epsilon_g/ell,epsilon_H/(2*LH))；固定步数 NesterovAGD，warm start y，重启动量 |
| Algorithm 1 Step 3 | 原始 grad_x f 和 Schur Hessian，不用 Newton 修正梯度 |
| (2.1) | 对称 dense eigh 求 [[H,g],[g.T,-alpha]] 最小特征对；检查特征残差 |
| (2.2) | abs(v)>=omega 使用 u/v，否则采用使 g.T@s 非正的符号 |
| (2.27) | 等价比较 norm(u)<Lambda*abs(v)，避免阈值接近1时舍入；返回完整步 x+s |
| 非终止更新 | s 归一化至长度 Lambda，无接受/拒绝机制 |

## 明确的实现选择

1. 首轮未知距离以初始残差/mu 上界代替，额外一次 grad_y 已计费。
2. N_t 使用原文系数 sqrt(kappa+1)，向上取整并截断到非负，允许0步；
   后续距离上界是 A+kappa*实际上一外层步长。这是满足 Lemma 2.3 下界的具体选择，
   未将固定步数内层改为残差早停。
3. 默认 omega=0.25，满足原文 (0,1/2)。要求 0<epsilon<=min(1,L2/2)。
4. 负曲率分支 g.T@u=0 时选 +u；不能使用 sign(0)=0 清掉方向。
5. eigensolver 为浮点 dense eigh，返回数值残差及通过状态，不宣称精确算术求解。
6. 迭代/内层步数预算超限、外层停滞或数值失败明确返回未收敛，不放宽判据。
7. 输出 y 是最后一次模型对应的内层点，未对终止 x+s 额外求解；history 明确标注。
   独立 evaluator 必须在输出 x 重新求解内层，现有运行脚本已如此处理。
8. 内层计数用实际 grad_y 包装器；N 步恰计 N 次查询，不查询终点梯度，residual=None。
   原生 converged 只代表 (2.27) 通过，不等同于公共精度达标。

## 使用

在项目根目录、已安装 torch 的 Python 环境运行：

```sh
python numerical_experiment/scripts/run_hsda.py --d 2 --epsilon 0.01 --max-iterations 2000
python numerical_experiment/scripts/run_comparision.py --algorithms utr5 hsda --d 2 --epsilon 0.001 --max-iterations 2000
python numerical_experiment/src_test/test_hsda.py
```

运行脚本沿用原有 runner；本轮未更改 UTR5、MCN、SCAR、AGD 的计算逻辑。
runner 历史 MCN 初始化成本排除规则尚未修改；正式跨方法比较前需按五算法设计统一成本口径。

## 已执行验证

- 最小特征对残差、单位长度及随机方向 Rayleigh 商比较。
- 原始 oracle 下的完整终止步与独立真实梯度检查。
- 零梯度负曲率点不产生零逃逸方向，非终止步长度恰为 Lambda。
- 可知 y* 的二次内层：初始和 warm-start 固定步数误差不超过 A。
- 内层查询计数逐项一致，预算耗尽不假报收敛。
- 多步二次原函数收敛及零梯度正曲率点的终止。
- 现有 CosLogCosh runner 的二维 seed=2、epsilon=0.01 小例：153次外层、2761次
  内层梯度，真实参考梯度范数约0.00955，Hessian 最小特征值约1.0257；公共判据通过。
- 同一二维问题 epsilon=0.001：HSDA 485次外层、10196次内层梯度，参考梯度范数
  约8.2054e-4，Hessian 最小特征值约1.02647；与 UTR5 共用入口均通过计数和公共精度检查。

这些验证用于实现验收，不是正式性能比较或复杂度实证。
