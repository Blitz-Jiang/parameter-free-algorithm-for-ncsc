# 六个算法的完整实现伪代码

对应当前工作区实现。以下每个算法均包含独立的主流程、内层求解和必要的数值实现说明。

## 目录

1. [UTR5-fixed（mpmath 80位，严格次数）](#algorithm-1)
2. [UTR5（float64，冻结总目标）](#algorithm-2)
3. [UTR5-early（float64，AR 内部早停）](#algorithm-3)
4. [GRTR（float64，理论参数版）](#algorithm-4)
5. [HSDA（float64，dense 特征子问题）](#algorithm-5)
6. [MCN（float64，残差上界初始化）](#algorithm-6)

---

<a id="algorithm-1"></a>

## UTR5-fixed（mpmath 80位，严格次数）

源码：algorithms/utr5_fixed_mp.py；inner_solvers/scar_mp.py；subproblem_solvers/trs_mp.py；problems/coslogcosh_mp.py。

本文对应 **2026-09-24 工作区当前源码**（包括未提交改动），描述实际实现而非理想化论文算法。
`epsilon` 均为方法内部参数；原生停止不等于统一外部精度。代码块中的实数运算采用本文件标注的数值后端。
每次真实的向量 `grad_y` 查询计作一次内层梯度；读缓存不计费。矩阵分解/特征求解不计入该计数。
每个算法章节包含自己的主流程和所需子程序。

### 导数与值构造

记 `q_x(y)=-f(x,y)`，`gy=grad_y f(x,y)`，Hessian 分块为 `Hxx,Hxy,Hyx,Hyy`。

```text
RAW(x,y):
    g = grad_x f(x,y)
    H = Hxx - Hxy * solve(Hyy,Hyx)
    return g, (H+H^T)/2

CORRECTED(x,y,缓存 gy, need_value):
    v = solve(Hyy,gy)
    g = grad_x f(x,y) - Hxy*v
    H = Hxx - Hxy*solve(Hyy,Hyx)
    H = (H+H^T)/2
    若 need_value: P_hat = f(x,y) - gy^T*v/2
    return P_hat（可省略）,g,H
```

RAW 不使用 Newton 修正梯度；CORRECTED 不重新查询已有 gy。各方法只调用主流程指定的构造器。

### 完整外层

```text
输入 x0,y_minus1,0<epsilon<1,外层预算 Qmax
c0=2^(-12); sigma=1/log(e/epsilon)
x=x0; y=y_minus1
查询 gy=grad_y f(x,y)
a0=SECANT(x,y,gy); state.nu=state.M=a0
(y,gy)=REFINE(x,y,gy,c0*a0*sqrt(epsilon)/(4*sigma),state)
(P_hat,g,H)=CORRECTED(x,y,gy,true)

for trial=1,...,Qmax:
    h=max(||g||,epsilon)
    R=sqrt(h)/(4*sigma); r_work=sqrt(epsilon)/(4*sigma)
    B=H+(sigma*sqrt(h)+sigma*sqrt(epsilon)/64)*I
    (d,lambda)=TR(g,B,R); r=||d||
    W=sigma*sqrt(h)*r^2; U=lambda*r^2; T=epsilon^(3/2)/sigma
    N=ceil(log2(1+r/(c0*r_work)))
    xp=x+d; yp=y
    gp=缓存 gy（候选与当前点相同）否则 grad_y f(xp,yp)
    [执行下面本版本的 TRACKING 块]

    若 h=epsilon 且 TR 判定为内部步:
        av=SECANT(xp,yp,gp)              # 冻结 av
        (yv,gv)=REFINE(xp,yp,gp,c0*av*epsilon^(3/2),state)
        (_,g_valid,H_valid)=CORRECTED(xp,yv,gv,false)
        若 ||g_valid||<=epsilon/2 且
           lambda_min(H_valid)>=-(21/8)*sigma*sqrt(epsilon):
            返回成功，x=xp,y=yv
        否则: reject=true
    否则:
        (Pp,gp_corrected,Hp)=CORRECTED(xp,yp,gp,true)
        若 P_hat-Pp >= W/8+U/4-T/1024 且
           ||gp_corrected||<=h/2+lambda*r:
            (x,y,gy,P_hat,g,H)=(xp,yp,gp,Pp,gp_corrected,Hp)
            sigma 不变；继续下一 trial
        否则: reject=true

    若 reject:
        丢弃候选/验证点，保留共享 state
        sigma=2*sigma
        (y,gy)=SCAR_HALF(x,原来存储的 y,原来 gy,state)
        (P_hat,g,H)=CORRECTED(x,y,gy,true)

预算耗尽/求解异常: 返回未收敛及失败原因，不放宽停止条件
```

#### TRACKING：严格调用次数

```text
for j=1,...,N:
    (yp,gp)=SCAR_HALF(xp,yp,gp,state)
```

即便提前达到了 2^(-N)*R_in，也不跳过剩余调用；零残差调用直接返回，仍记作完成一次调用。
保存 tracking_halvings 与 working_halvings_completed，成功运行应逐 trial 相等。

### 普通 AR 与 persistent SCARHalf

以下 `rho_q(v,z)=q(z)-q(v)-grad q(v)^T(z-v)` 用稳定函数余项计算，避免直接相减消去。
`a` 表示 AR 正则化参数，与外层 sigma 不同。

```text
AR_SUBPROBLEM(x,u,bar,a,L,gq_cached):
    y=v=u; theta=1
    for k=1,...,100000:
        gq = gq_cached（仅 k=1 且缓存可用）否则 -grad_y f(x,v)
        repeat:                         # 至多100次回溯
            z = v + (-gq + a*(bar-v))/(L+a)
            若 rho_q(v,z) <= L*||z-v||^2/2: break
            L = 2*L
        若 k >= ceil(8*sqrt(4*L/a)): return z
        theta_new = (1+sqrt(1+4*theta^2))/2
        v = z + (theta-1)/theta_new*(z-y)
        y=z; theta=theta_new
    预算耗尽: 报告失败

AR_BACKTRACK(x,z,bar,a,M):
    gq = -grad_y f(x,z)
    gs = gq + a*(z-bar)
    repeat:                             # 至多100次
        w = z - gs/(2*(M+a))
        若 rho_q(z,w) <= M*||w-z||^2/2: return M,gq
        M = 2*M
    预算耗尽: 报告失败

AR(x,u,缓存 gy,nu,M):
    y=bar=u; a_prev=0; a=nu/10; gq=-gy
    repeat:
        gamma = 1-a_prev/a
        bar = (1-gamma)*bar + gamma*y
        z = AR_SUBPROBLEM(x,y,bar,a,M/2,gq)
        (M_new,gq_new) = AR_BACKTRACK(x,z,bar,a,M/2)
        若 a >= M_new: return z,-gq_new,M_new
        y=z; gq=gq_new; M=M_new; a_prev=a; a=4*a

SCAR_HALF(x,u,缓存 gy,共享 state={nu,M}):
    冻结 R=||gy||
    若 R=0: return u,gy                  # 已精确为零，不查询梯度
    repeat:
        (z,gz,M_new) = AR(x,u,gy,nu,M)
        state.M = M_new                  # 即便失败也保留
        若 ||gz||<=R/2: return z,gz      # 成功减半
        state.nu = nu/4
        保留原始 u,gy,R，重试             # 不从失败 z 继续

SECANT(x,y,缓存 gy):
    z = y + ones(dim_y)/sqrt(dim_y)
    return ||grad_y f(x,z)-gy||/||z-y||  # 额外一次查询

REFINE(x,y,缓存 gy,tau,state):
    while ||gy||>tau:
        (y,gy)=SCAR_HALF(x,y,gy,state)
    return y,gy
```

SCAR 的曲率校准跨候选、拒绝、验证保留，但每次 AR 的动量重新初始化。
float64 默认 SCARHalf 无尝试数上限（可选配置上限）；AR 的非有限数、下溢、回溯/子问题预算失败均报告错误。
mpmath 版默认每次 SCARHalf 最多8次 AR 尝试、每次 AR 最多100阶段；失败不算成功减半。

### Dense TR 子程序

目标为 `min_{||s||<=R} g^T s + s^T B s/2`。

```text
DENSE_TR(g,B,R):
    对称化 B；[theta,V] = eigh(B)，a = V^T*g
    若 B 数值半正定且 a 在零空间的分量足够小:
        在正特征值方向取 z_i=-a_i/theta_i，零空间取0
        若 ||z||<=R: return V*z, lambda=0
    lambda_low = max(0,-theta_min)
    若 lambda_low>0 且 a 在 B+lambda_low*I 的零空间分量足够小:
        非零空间取 z_i=-a_i/(theta_i+lambda_low)
        若 ||z||<=R:
            在最小特征向量方向补足 sqrt(R^2-||z||^2)
            return V*z,lambda_low                 # hard case
    寻找 lambda>lambda_low，使
        ||(-a_i/(theta_i+lambda))_i|| = R
    倍增右端点括住根，再二分；return s=V*z,lambda
```

float64 基础求解器使用绝对半径残差容差，默认 tol=1e-14、二分最多1000次；
零特征值阈值为 max(tol,1e-14)，普通求根左端点为 lambda_low+该阈值。
未收敛返回 converged=False 或抛出异常；各外层如何验证见其主流程后的说明。

### mpmath 后端的具体实现

默认80位十进制有效数字，独立 context，最低40位。x/y、动量、nu/M、oracle、特征求解均保留高精度。
只有输入既有浮点系数和最终供现有 evaluator 使用的输出作转换；history 另存 x_mp/y_mp。
不使用 AR early，不把浮点残差平台强制置零；尝试预算耗尽返回失败。

当前显式后端仅支持 canonical CosLogCosh：

```text
f(x,y) = sum_i (cos(omega*x_i)-1)/omega²
         + a*x^T Q*y - mu_y*||y||²/2 - beta*sum_j log(cosh(y_j))
a = 既有问题对象的 a_mu（0.5*sqrt(mu_y) 的已存储系数）
gy = a*Q^T*x - mu_y*y - beta*tanh(y)
gx = -sin(omega*x)/omega + a*Q*y
Hxx = diag(-cos(omega*x))
Hxy = a*Q; Hyx = Hxy^T
Hyy = diag(-mu_y-beta*(1-tanh(y)²))
```

修正 oracle 按前面的公式计算；原函数值里的 cos(t)-1 使用 -2*sin(t/2)²。
内层稳定余项令 d=z-y，A=±2d（y>=0 时取负），p=1/(1+exp(2|y|))，w=p*expm1(A)：
`q_remainder=sum[mu_y*d²/2+beta*(log1p(w)-p*A)]`。
小 A 时改写成 `p*(expm1(A)-A)+(log1p(w)-w)`，两部分分别从二次项求级数到 context 精度，避免消去。

mpmath TR 使用 DENSE_TR 的谱方法，tol=10^(-min(40,dps//2))；零空间阈值随矩阵范数缩放，
直接在原模型求解，右端点最多倍增1000次，二分最多8*dps+100次，相对半径误差<=tol。
额外验证 KKT，容差100*tol；边界判据为 lambda>0 或相对半径差<=100*tol。
它不是 float64 包装器，也不先转成 torch 求解。

每次 AR_SUBPROBLEM/AR_BACKTRACK 的回溯实际调用 f 的次数由 MP 后端计数，
不能假设与 torch/autograd 的函数调用数完全相同；向量内层梯度的计数单位一致。
不同精度/解析后端的 wall time 不适合直接作公平速度排名。

### 返回与日志

返回 x/y、converged、失败原因、外层 trial 数、实际内层梯度数，以及接受/拒绝/验证、减半次数和状态日志。
终止验证返回候选 xp；失败刷新始终从旧 x/y 开始。独立终点评价重新求解输出 x 的内层，不复用算法的误差声明。

---

<a id="algorithm-2"></a>

## UTR5（float64，冻结总目标）

源码：algorithms/utr5.py；继承 algorithms/utr4.py、utr3.py；inner_solvers/scar.py。

本文对应 **2026-09-24 工作区当前源码**（包括未提交改动），描述实际实现而非理想化论文算法。
`epsilon` 均为方法内部参数；原生停止不等于统一外部精度。代码块中的实数运算采用本文件标注的数值后端。
每次真实的向量 `grad_y` 查询计作一次内层梯度；读缓存不计费。矩阵分解/特征求解不计入该计数。
每个算法章节包含自己的主流程和所需子程序。

### 导数与值构造

记 `q_x(y)=-f(x,y)`，`gy=grad_y f(x,y)`，Hessian 分块为 `Hxx,Hxy,Hyx,Hyy`。

```text
RAW(x,y):
    g = grad_x f(x,y)
    H = Hxx - Hxy * solve(Hyy,Hyx)
    return g, (H+H^T)/2

CORRECTED(x,y,缓存 gy, need_value):
    v = solve(Hyy,gy)
    g = grad_x f(x,y) - Hxy*v
    H = Hxx - Hxy*solve(Hyy,Hyx)
    H = (H+H^T)/2
    若 need_value: P_hat = f(x,y) - gy^T*v/2
    return P_hat（可省略）,g,H
```

RAW 不使用 Newton 修正梯度；CORRECTED 不重新查询已有 gy。各方法只调用主流程指定的构造器。

### 完整外层

```text
输入 x0,y_minus1,0<epsilon<1,外层预算 Qmax
c0=2^(-12); sigma=1/log(e/epsilon)
x=x0; y=y_minus1
查询 gy=grad_y f(x,y)
a0=SECANT(x,y,gy); state.nu=state.M=a0
(y,gy)=REFINE(x,y,gy,c0*a0*sqrt(epsilon)/(4*sigma),state)
(P_hat,g,H)=CORRECTED(x,y,gy,true)

for trial=1,...,Qmax:
    h=max(||g||,epsilon)
    R=sqrt(h)/(4*sigma); r_work=sqrt(epsilon)/(4*sigma)
    B=H+(sigma*sqrt(h)+sigma*sqrt(epsilon)/64)*I
    (d,lambda)=TR(g,B,R); r=||d||
    W=sigma*sqrt(h)*r^2; U=lambda*r^2; T=epsilon^(3/2)/sigma
    N=ceil(log2(1+r/(c0*r_work)))
    xp=x+d; yp=y
    gp=缓存 gy（候选与当前点相同）否则 grad_y f(xp,yp)
    [执行下面本版本的 TRACKING 块]

    若 h=epsilon 且 TR 判定为内部步:
        av=SECANT(xp,yp,gp)              # 冻结 av
        (yv,gv)=REFINE(xp,yp,gp,c0*av*epsilon^(3/2),state)
        (_,g_valid,H_valid)=CORRECTED(xp,yv,gv,false)
        若 ||g_valid||<=epsilon/2 且
           lambda_min(H_valid)>=-(21/8)*sigma*sqrt(epsilon):
            返回成功，x=xp,y=yv
        否则: reject=true
    否则:
        (Pp,gp_corrected,Hp)=CORRECTED(xp,yp,gp,true)
        若 P_hat-Pp >= W/8+U/4-T/1024 且
           ||gp_corrected||<=h/2+lambda*r:
            (x,y,gy,P_hat,g,H)=(xp,yp,gp,Pp,gp_corrected,Hp)
            sigma 不变；继续下一 trial
        否则: reject=true

    若 reject:
        丢弃候选/验证点，保留共享 state
        sigma=2*sigma
        (y,gy)=SCAR_HALF(x,原来存储的 y,原来 gy,state)
        (P_hat,g,H)=CORRECTED(x,y,gy,true)

预算耗尽/求解异常: 返回未收敛及失败原因，不放宽停止条件
```

#### TRACKING：冻结总残差目标

```text
冻结 R_in=||gp||
tau_track=2^(-N)*R_in
for j=1,...,N:
    若 ||gp||<=tau_track: break
    (yp,gp)=SCAR_HALF(xp,yp,gp,state)
```

该优化只发生在外层工作跟踪；每个 SCARHalf 自己仍须成功减半。不能把它与 AR 内部 early 混为一谈。

### 普通 AR 与 persistent SCARHalf

以下 `rho_q(v,z)=q(z)-q(v)-grad q(v)^T(z-v)` 用稳定函数余项计算，避免直接相减消去。
`a` 表示 AR 正则化参数，与外层 sigma 不同。

```text
AR_SUBPROBLEM(x,u,bar,a,L,gq_cached):
    y=v=u; theta=1
    for k=1,...,100000:
        gq = gq_cached（仅 k=1 且缓存可用）否则 -grad_y f(x,v)
        repeat:                         # 至多100次回溯
            z = v + (-gq + a*(bar-v))/(L+a)
            若 rho_q(v,z) <= L*||z-v||^2/2: break
            L = 2*L
        若 k >= ceil(8*sqrt(4*L/a)): return z
        theta_new = (1+sqrt(1+4*theta^2))/2
        v = z + (theta-1)/theta_new*(z-y)
        y=z; theta=theta_new
    预算耗尽: 报告失败

AR_BACKTRACK(x,z,bar,a,M):
    gq = -grad_y f(x,z)
    gs = gq + a*(z-bar)
    repeat:                             # 至多100次
        w = z - gs/(2*(M+a))
        若 rho_q(z,w) <= M*||w-z||^2/2: return M,gq
        M = 2*M
    预算耗尽: 报告失败

AR(x,u,缓存 gy,nu,M):
    y=bar=u; a_prev=0; a=nu/10; gq=-gy
    repeat:
        gamma = 1-a_prev/a
        bar = (1-gamma)*bar + gamma*y
        z = AR_SUBPROBLEM(x,y,bar,a,M/2,gq)
        (M_new,gq_new) = AR_BACKTRACK(x,z,bar,a,M/2)
        若 a >= M_new: return z,-gq_new,M_new
        y=z; gq=gq_new; M=M_new; a_prev=a; a=4*a

SCAR_HALF(x,u,缓存 gy,共享 state={nu,M}):
    冻结 R=||gy||
    若 R=0: return u,gy                  # 已精确为零，不查询梯度
    repeat:
        (z,gz,M_new) = AR(x,u,gy,nu,M)
        state.M = M_new                  # 即便失败也保留
        若 ||gz||<=R/2: return z,gz      # 成功减半
        state.nu = nu/4
        保留原始 u,gy,R，重试             # 不从失败 z 继续

SECANT(x,y,缓存 gy):
    z = y + ones(dim_y)/sqrt(dim_y)
    return ||grad_y f(x,z)-gy||/||z-y||  # 额外一次查询

REFINE(x,y,缓存 gy,tau,state):
    while ||gy||>tau:
        (y,gy)=SCAR_HALF(x,y,gy,state)
    return y,gy
```

SCAR 的曲率校准跨候选、拒绝、验证保留，但每次 AR 的动量重新初始化。
float64 默认 SCARHalf 无尝试数上限（可选配置上限）；AR 的非有限数、下溢、回溯/子问题预算失败均报告错误。
mpmath 版默认每次 SCARHalf 最多8次 AR 尝试、每次 AR 最多100阶段；失败不算成功减半。

### Dense TR 子程序

目标为 `min_{||s||<=R} g^T s + s^T B s/2`。

```text
DENSE_TR(g,B,R):
    对称化 B；[theta,V] = eigh(B)，a = V^T*g
    若 B 数值半正定且 a 在零空间的分量足够小:
        在正特征值方向取 z_i=-a_i/theta_i，零空间取0
        若 ||z||<=R: return V*z, lambda=0
    lambda_low = max(0,-theta_min)
    若 lambda_low>0 且 a 在 B+lambda_low*I 的零空间分量足够小:
        非零空间取 z_i=-a_i/(theta_i+lambda_low)
        若 ||z||<=R:
            在最小特征向量方向补足 sqrt(R^2-||z||^2)
            return V*z,lambda_low                 # hard case
    寻找 lambda>lambda_low，使
        ||(-a_i/(theta_i+lambda))_i|| = R
    倍增右端点括住根，再二分；return s=V*z,lambda
```

float64 基础求解器使用绝对半径残差容差，默认 tol=1e-14、二分最多1000次；
零特征值阈值为 max(tol,1e-14)，普通求根左端点为 lambda_low+该阈值。
未收敛返回 converged=False 或抛出异常；各外层如何验证见其主流程后的说明。

### float64 TR 包装与计数

UTR5 不直接用裸 DENSE_TR：先设 `S=max(||R*g||,||R²*B||_F)`，
将模型变成 `g_unit=R*g/S, B_unit=R²*B/S, radius=1`。
令 `eta=max(1000*machine_epsilon,用户tol)`，复制求解器并设 tol=eta/4。
恢复 `d=R*s_unit, lambda=lambda_unit*S/R²` 后检查有限性、半径、站立性、半正定性和互补性；
边界判据为 `abs(||d||-R)<=eta*R`。

若原求解器失败或 KKT 检查失败，再在特征空间使用 `lambda=lambda_low+delta` 的偏移求根：
零空间条件满足时直接处理 hard case，否则在 `delta∈[0,||g_unit||]` 二分至单位半径误差<=eta/4。
达到机器分辨率/迭代上限仍不满足则报错，不静默接受。
修复调用另记 tr_refined；当前 n_subproblem_solves 统计外层 trial 数，不把该数值修复另算一个外层模型。

稳定 LogCosh 包装用于内层和 corrected oracle，曲率/验证检查保留浮点容差。
所有真实 grad_y 查询逐阶段计数；终点 gy 可复用。相同候选用 torch.equal 判定，可复用缓存。

### 返回与日志

返回 x/y、converged、失败原因、外层 trial 数、实际内层梯度数，以及接受/拒绝/验证、减半次数和状态日志。
终止验证返回候选 xp；失败刷新始终从旧 x/y 开始。独立终点评价重新求解输出 x 的内层，不复用算法的误差声明。

---

<a id="algorithm-3"></a>

## UTR5-early（float64，AR 内部早停）

源码：algorithms/utr5.py；inner_solvers/scar_persistent_early.py；inner_solvers/scar.py。

本文对应 **2026-09-24 工作区当前源码**（包括未提交改动），描述实际实现而非理想化论文算法。
`epsilon` 均为方法内部参数；原生停止不等于统一外部精度。代码块中的实数运算采用本文件标注的数值后端。
每次真实的向量 `grad_y` 查询计作一次内层梯度；读缓存不计费。矩阵分解/特征求解不计入该计数。
每个算法章节包含自己的主流程和所需子程序。

### 导数与值构造

记 `q_x(y)=-f(x,y)`，`gy=grad_y f(x,y)`，Hessian 分块为 `Hxx,Hxy,Hyx,Hyy`。

```text
RAW(x,y):
    g = grad_x f(x,y)
    H = Hxx - Hxy * solve(Hyy,Hyx)
    return g, (H+H^T)/2

CORRECTED(x,y,缓存 gy, need_value):
    v = solve(Hyy,gy)
    g = grad_x f(x,y) - Hxy*v
    H = Hxx - Hxy*solve(Hyy,Hyx)
    H = (H+H^T)/2
    若 need_value: P_hat = f(x,y) - gy^T*v/2
    return P_hat（可省略）,g,H
```

RAW 不使用 Newton 修正梯度；CORRECTED 不重新查询已有 gy。各方法只调用主流程指定的构造器。

### 完整外层

```text
输入 x0,y_minus1,0<epsilon<1,外层预算 Qmax
c0=2^(-12); sigma=1/log(e/epsilon)
x=x0; y=y_minus1
查询 gy=grad_y f(x,y)
a0=SECANT(x,y,gy); state.nu=state.M=a0
(y,gy)=REFINE(x,y,gy,c0*a0*sqrt(epsilon)/(4*sigma),state)
(P_hat,g,H)=CORRECTED(x,y,gy,true)

for trial=1,...,Qmax:
    h=max(||g||,epsilon)
    R=sqrt(h)/(4*sigma); r_work=sqrt(epsilon)/(4*sigma)
    B=H+(sigma*sqrt(h)+sigma*sqrt(epsilon)/64)*I
    (d,lambda)=TR(g,B,R); r=||d||
    W=sigma*sqrt(h)*r^2; U=lambda*r^2; T=epsilon^(3/2)/sigma
    N=ceil(log2(1+r/(c0*r_work)))
    xp=x+d; yp=y
    gp=缓存 gy（候选与当前点相同）否则 grad_y f(xp,yp)
    [执行下面本版本的 TRACKING 块]

    若 h=epsilon 且 TR 判定为内部步:
        av=SECANT(xp,yp,gp)              # 冻结 av
        (yv,gv)=REFINE(xp,yp,gp,c0*av*epsilon^(3/2),state)
        (_,g_valid,H_valid)=CORRECTED(xp,yv,gv,false)
        若 ||g_valid||<=epsilon/2 且
           lambda_min(H_valid)>=-(21/8)*sigma*sqrt(epsilon):
            返回成功，x=xp,y=yv
        否则: reject=true
    否则:
        (Pp,gp_corrected,Hp)=CORRECTED(xp,yp,gp,true)
        若 P_hat-Pp >= W/8+U/4-T/1024 且
           ||gp_corrected||<=h/2+lambda*r:
            (x,y,gy,P_hat,g,H)=(xp,yp,gp,Pp,gp_corrected,Hp)
            sigma 不变；继续下一 trial
        否则: reject=true

    若 reject:
        丢弃候选/验证点，保留共享 state
        sigma=2*sigma
        (y,gy)=SCAR_HALF(x,原来存储的 y,原来 gy,state)
        (P_hat,g,H)=CORRECTED(x,y,gy,true)

预算耗尽/求解异常: 返回未收敛及失败原因，不放宽停止条件
```

#### TRACKING：冻结总残差目标

```text
冻结 R_in=||gp||
tau_track=2^(-N)*R_in
for j=1,...,N:
    若 ||gp||<=tau_track: break
    (yp,gp)=SCAR_HALF(xp,yp,gp,state)
```

该优化只发生在外层工作跟踪；每个 SCARHalf 自己仍须成功减半。不能把它与 AR 内部 early 混为一谈。

### 普通 AR 与 persistent SCARHalf

以下 `rho_q(v,z)=q(z)-q(v)-grad q(v)^T(z-v)` 用稳定函数余项计算，避免直接相减消去。
`a` 表示 AR 正则化参数，与外层 sigma 不同。

```text
AR_SUBPROBLEM(x,u,bar,a,L,gq_cached):
    y=v=u; theta=1
    for k=1,...,100000:
        gq = gq_cached（仅 k=1 且缓存可用）否则 -grad_y f(x,v)
        repeat:                         # 至多100次回溯
            z = v + (-gq + a*(bar-v))/(L+a)
            若 rho_q(v,z) <= L*||z-v||^2/2: break
            L = 2*L
        若 k >= ceil(8*sqrt(4*L/a)): return z
        theta_new = (1+sqrt(1+4*theta^2))/2
        v = z + (theta-1)/theta_new*(z-y)
        y=z; theta=theta_new
    预算耗尽: 报告失败

AR_BACKTRACK(x,z,bar,a,M):
    gq = -grad_y f(x,z)
    gs = gq + a*(z-bar)
    repeat:                             # 至多100次
        w = z - gs/(2*(M+a))
        若 rho_q(z,w) <= M*||w-z||^2/2: return M,gq
        M = 2*M
    预算耗尽: 报告失败

AR(x,u,缓存 gy,nu,M):
    y=bar=u; a_prev=0; a=nu/10; gq=-gy
    repeat:
        gamma = 1-a_prev/a
        bar = (1-gamma)*bar + gamma*y
        z = AR_SUBPROBLEM(x,y,bar,a,M/2,gq)
        (M_new,gq_new) = AR_BACKTRACK(x,z,bar,a,M/2)
        若 a >= M_new: return z,-gq_new,M_new
        y=z; gq=gq_new; M=M_new; a_prev=a; a=4*a

SCAR_HALF(x,u,缓存 gy,共享 state={nu,M}):
    冻结 R=||gy||
    若 R=0: return u,gy                  # 已精确为零，不查询梯度
    repeat:
        (z,gz,M_new) = AR(x,u,gy,nu,M)
        state.M = M_new                  # 即便失败也保留
        若 ||gz||<=R/2: return z,gz      # 成功减半
        state.nu = nu/4
        保留原始 u,gy,R，重试             # 不从失败 z 继续

SECANT(x,y,缓存 gy):
    z = y + ones(dim_y)/sqrt(dim_y)
    return ||grad_y f(x,z)-gy||/||z-y||  # 额外一次查询

REFINE(x,y,缓存 gy,tau,state):
    while ||gy||>tau:
        (y,gy)=SCAR_HALF(x,y,gy,state)
    return y,gy
```

SCAR 的曲率校准跨候选、拒绝、验证保留，但每次 AR 的动量重新初始化。
float64 默认 SCARHalf 无尝试数上限（可选配置上限）；AR 的非有限数、下溢、回溯/子问题预算失败均报告错误。
mpmath 版默认每次 SCARHalf 最多8次 AR 尝试、每次 AR 最多100阶段；失败不算成功减半。

### AR 内部 early 机制（本文件的 SCAR_HALF 调用此 AR）

上述普通 AR 的方程保持不变，但每一次进入 AR 时安装以下梯度查询拦截器：

```text
EARLY_AR(x,u,gy,nu,M):
    冻结 threshold=||gy||/2
    last_M=M
    若 ||gy||=0: return u,gy,M
    在本次 AR 的每次真实 grad_y(x,z) 查询上:
        gz=真正查询的梯度；计数加1
        若 ||gz||<=threshold:
            立即中断 AR，返回该查询的 z,gz,last_M
        否则向 AR 提供 gz
    每次 AR_BACKTRACK 完整成功后:
        last_M=该次返回 M
    若没有提前中断:
        按普通 AR 的 a>=M 条件返回
```

提前中断可以发生在加速子问题的梯度查询或回溯前的梯度查询。
只保留入口/最近一次完整成功回溯的 M，不保留尚未通过回溯的临时值。
若完整 AR 没实现减半，SCAR_HALF 仍将 nu 除4并重试；它不是每次无条件接受的内层算法。
计数包含触发 early 的那次梯度查询，缓存入口梯度不重复计数。

### Dense TR 子程序

目标为 `min_{||s||<=R} g^T s + s^T B s/2`。

```text
DENSE_TR(g,B,R):
    对称化 B；[theta,V] = eigh(B)，a = V^T*g
    若 B 数值半正定且 a 在零空间的分量足够小:
        在正特征值方向取 z_i=-a_i/theta_i，零空间取0
        若 ||z||<=R: return V*z, lambda=0
    lambda_low = max(0,-theta_min)
    若 lambda_low>0 且 a 在 B+lambda_low*I 的零空间分量足够小:
        非零空间取 z_i=-a_i/(theta_i+lambda_low)
        若 ||z||<=R:
            在最小特征向量方向补足 sqrt(R^2-||z||^2)
            return V*z,lambda_low                 # hard case
    寻找 lambda>lambda_low，使
        ||(-a_i/(theta_i+lambda))_i|| = R
    倍增右端点括住根，再二分；return s=V*z,lambda
```

float64 基础求解器使用绝对半径残差容差，默认 tol=1e-14、二分最多1000次；
零特征值阈值为 max(tol,1e-14)，普通求根左端点为 lambda_low+该阈值。
未收敛返回 converged=False 或抛出异常；各外层如何验证见其主流程后的说明。

### float64 TR 包装与计数

UTR5 不直接用裸 DENSE_TR：先设 `S=max(||R*g||,||R²*B||_F)`，
将模型变成 `g_unit=R*g/S, B_unit=R²*B/S, radius=1`。
令 `eta=max(1000*machine_epsilon,用户tol)`，复制求解器并设 tol=eta/4。
恢复 `d=R*s_unit, lambda=lambda_unit*S/R²` 后检查有限性、半径、站立性、半正定性和互补性；
边界判据为 `abs(||d||-R)<=eta*R`。

若原求解器失败或 KKT 检查失败，再在特征空间使用 `lambda=lambda_low+delta` 的偏移求根：
零空间条件满足时直接处理 hard case，否则在 `delta∈[0,||g_unit||]` 二分至单位半径误差<=eta/4。
达到机器分辨率/迭代上限仍不满足则报错，不静默接受。
修复调用另记 tr_refined；当前 n_subproblem_solves 统计外层 trial 数，不把该数值修复另算一个外层模型。

稳定 LogCosh 包装用于内层和 corrected oracle，曲率/验证检查保留浮点容差。
所有真实 grad_y 查询逐阶段计数；终点 gy 可复用。相同候选用 torch.equal 判定，可复用缓存。

### 返回与日志

返回 x/y、converged、失败原因、外层 trial 数、实际内层梯度数，以及接受/拒绝/验证、减半次数和状态日志。
终止验证返回候选 xp；失败刷新始终从旧 x/y 开始。独立终点评价重新求解输出 x 的内层，不复用算法的误差声明。

---

<a id="algorithm-4"></a>

## GRTR（float64，理论参数版）

源码：algorithms/grtr.py；inner_solvers/nesterov.py；subproblem_solvers/trs.py。

本文对应 **2026-09-24 工作区当前源码**（包括未提交改动），描述实际实现而非理想化论文算法。
`epsilon` 均为方法内部参数；原生停止不等于统一外部精度。代码块中的实数运算采用本文件标注的数值后端。
每次真实的向量 `grad_y` 查询计作一次内层梯度；读缓存不计费。矩阵分解/特征求解不计入该计数。
每个算法章节包含自己的主流程和所需子程序。

### 参数与内层次数

```text
kappa=ell/mu; L1=(1+kappa)*ell
LH=rho*(1+kappa)^2; L2=rho*(1+kappa)^3
输入已知的 ell,mu,rho（从 problem 取得）
COUNT(D):
    若 D<=0: return 0
    return max(0,ceil(2*sqrt(kappa)*log(sqrt(kappa+1)*D/A)))
```

COUNT 使用论文的 sqrt(kappa+1) 前因子；允许零步，不额外强制至少一步。
首轮以 residual/mu 代替未知距离，初始化查询单独计费。若所需 COUNT 超过 max_inner_steps，报告预算失败，不能截断后继续。

### GRTR 主流程

```text
输入 x0,y_minus1,epsilon∈(0,1],TR预算 Qmax
sigma=sqrt(L2)/2; radius_scale=1/(4*sqrt(L2))
epsilon_g=min(1/96,sqrt(L2)/(16*L1))*epsilon^(3/2)
epsilon_H=sqrt(L2*epsilon)/12
A=min(epsilon_g/ell,epsilon_H/(2*LH))
x=x0; y=y_minus1
D=||grad_y f(x,y)||/mu                 # 初始化查询计费
for t=0,...,Qmax-1:
    N=COUNT(D); y=AGD(x,y,N)
    核对内层实际查询数、N 与返回计数，检查有限性
    (g,H)=RAW(x,y)
    R=radius_scale*sqrt(max(||g||,epsilon))
    B=H+sigma*sqrt(||g||)*I           # 正则项无 epsilon 截断
    (s,lambda)=DENSE_TR(g,B,R)
    验证求解成功及 KKT；不满足则失败
    若 ||g||<=epsilon 且 lambda<=sqrt(L2*epsilon):
        return 成功，当前 x,y          # 不执行 s
    若 x+s 非有限或在浮点下等于 x: return 失败
    x=x+s
    D=A+kappa*||s||
预算耗尽: return 当前点，未收敛
```

### 数值验证与结果

KKT 归一化容差默认 max(1e-8,64*machine_epsilon)：
站立性以 max(||g||+(||B||_2+|lambda|)||s||,1e-30) 归一化；
可行性以 R 归一化；互补性以 max(|lambda|R,1e-30) 归一化；
负曲率误差以 max(1,||B||_2+|lambda|) 归一化；负乘子误差以 max(1,|lambda|) 归一化。
任何一项超阈值返回失败，不像 UTR5 那样自动重做 TR。

停止是 Lemma 2.6 判据；在相应理论条件下对应真实梯度<=97*epsilon/96、
真实最小特征值>=-(19/12)*sqrt(L2*epsilon)，不是严格公共 epsilon 的同义词。
外层计数含终止用的一次 TR 求解，n_updates 单列；正常终止 y 属于当前 x，预算结束于更新后则可能属于旧 x。

### 导数与值构造

记 `q_x(y)=-f(x,y)`，`gy=grad_y f(x,y)`，Hessian 分块为 `Hxx,Hxy,Hyx,Hyy`。

```text
RAW(x,y):
    g = grad_x f(x,y)
    H = Hxx - Hxy * solve(Hyy,Hyx)
    return g, (H+H^T)/2

CORRECTED(x,y,缓存 gy, need_value):
    v = solve(Hyy,gy)
    g = grad_x f(x,y) - Hxy*v
    H = Hxx - Hxy*solve(Hyy,Hyx)
    H = (H+H^T)/2
    若 need_value: P_hat = f(x,y) - gy^T*v/2
    return P_hat（可省略）,g,H
```

RAW 不使用 Newton 修正梯度；CORRECTED 不重新查询已有 gy。各方法只调用主流程指定的构造器。

### 固定步数 Nesterov 内层

```text
AGD(x,y_in,N; ell,mu):
    kappa = ell/mu
    beta = (sqrt(kappa)-1)/(sqrt(kappa)+1)
    y = v = y_in                         # 每次调用重启动量
    for j=1,...,N:
        gv = grad_y f(x,v)               # 一次真实查询
        y_new = v + gv/ell
        v = y_new + beta*(y_new-y)
        y = y_new
    return y,N 次梯度查询                # N=0 时无查询，返回入口 y
```

固定步数分支不查询终点梯度，返回 grad_y=None、residual=None；converged=True 仅表示指定步数完成。
固定 N 的有效性来自外层误差预算。残差停止模式保持不变，仍测量并计入每次残差查询。

### Dense TR 子程序

目标为 `min_{||s||<=R} g^T s + s^T B s/2`。

```text
DENSE_TR(g,B,R):
    对称化 B；[theta,V] = eigh(B)，a = V^T*g
    若 B 数值半正定且 a 在零空间的分量足够小:
        在正特征值方向取 z_i=-a_i/theta_i，零空间取0
        若 ||z||<=R: return V*z, lambda=0
    lambda_low = max(0,-theta_min)
    若 lambda_low>0 且 a 在 B+lambda_low*I 的零空间分量足够小:
        非零空间取 z_i=-a_i/(theta_i+lambda_low)
        若 ||z||<=R:
            在最小特征向量方向补足 sqrt(R^2-||z||^2)
            return V*z,lambda_low                 # hard case
    寻找 lambda>lambda_low，使
        ||(-a_i/(theta_i+lambda))_i|| = R
    倍增右端点括住根，再二分；return s=V*z,lambda
```

float64 基础求解器使用绝对半径残差容差，默认 tol=1e-14、二分最多1000次；
零特征值阈值为 max(tol,1e-14)，普通求根左端点为 lambda_low+该阈值。
未收敛返回 converged=False 或抛出异常；各外层如何验证见其主流程后的说明。

---

<a id="algorithm-5"></a>

## HSDA（float64，dense 特征子问题）

源码：algorithms/hsda.py；subproblem_solvers/homogeneous.py；inner_solvers/nesterov.py。

本文对应 **2026-09-24 工作区当前源码**（包括未提交改动），描述实际实现而非理想化论文算法。
`epsilon` 均为方法内部参数；原生停止不等于统一外部精度。代码块中的实数运算采用本文件标注的数值后端。
每次真实的向量 `grad_y` 查询计作一次内层梯度；读缓存不计费。矩阵分解/特征求解不计入该计数。
每个算法章节包含自己的主流程和所需子程序。

### 参数与内层次数

```text
kappa=ell/mu; L1=(1+kappa)*ell
LH=rho*(1+kappa)^2; L2=rho*(1+kappa)^3
输入已知的 ell,mu,rho（从 problem 取得）
COUNT(D):
    若 D<=0: return 0
    return max(0,ceil(2*sqrt(kappa)*log(sqrt(kappa+1)*D/A)))
```

COUNT 使用论文的 sqrt(kappa+1) 前因子；允许零步，不额外强制至少一步。
首轮以 residual/mu 代替未知距离，初始化查询单独计费。若所需 COUNT 超过 max_inner_steps，报告预算失败，不能截断后继续。

### HSDA 主流程

```text
输入 x0,y_minus1,0<epsilon<=min(1,L2/2),omega∈(0,1/2)（默认0.25）
alpha=sqrt(L2*epsilon); Lambda=sqrt(epsilon/L2)
epsilon_g=epsilon/12; epsilon_H=alpha/12
A=min(epsilon_g/ell,epsilon_H/(2*LH))
x=x0; y=y_minus1; D=||grad_y f(x,y)||/mu
for t=1,...,Qmax:
    N=COUNT(D); y=AGD(x,y,N)
    核对内层步数和真实查询数，检查有限性
    (g,H)=RAW(x,y)
    G=[[H,g],[g^T,-alpha]]
    用对称 eigh 求 G 的最小特征值 theta 及单位特征向量 [u;v]
    将绝对值最大分量选为非负，固定特征向量符号
    若 ||G*[u;v]-theta*[u;v]|| > max(tol,32*machine_epsilon)*max(1,||G||_F): 失败
    若 |v|>=omega:
        s=u/v
        terminal = (||u|| < Lambda*|v|)
    否则:
        s=u（g^T*u<=0）否则 -u       # g^T*u=0 时也不清掉方向
        terminal=false
    若方向非有限或非终止方向为0: 失败
    d=s（terminal）否则 Lambda*s/||s||
    若 x+d 非有限或非终止步数值停滞: 失败
    x=x+d
    若 terminal: return 成功，x,y
    D=A+kappa*||d||
预算耗尽: return 当前点，未收敛
```

`||u||<Lambda*|v|` 与单位特征向量的原文 `|v|>1/sqrt(1+Lambda²)` 等价，避免比较接近1的阈值。
终止步是完整 s，其他步长度固定为 Lambda。实现的是 dense HSDA，不是 Lanczos IHSDA。
返回 y 对应终止更新前的模型点；独立评价器必须在输出 x 重新求解内层。
计数包括终止的特征子问题。支持独立 oracle 精度评价，原生停止不自动保证严格公共 epsilon。

### 导数与值构造

记 `q_x(y)=-f(x,y)`，`gy=grad_y f(x,y)`，Hessian 分块为 `Hxx,Hxy,Hyx,Hyy`。

```text
RAW(x,y):
    g = grad_x f(x,y)
    H = Hxx - Hxy * solve(Hyy,Hyx)
    return g, (H+H^T)/2

CORRECTED(x,y,缓存 gy, need_value):
    v = solve(Hyy,gy)
    g = grad_x f(x,y) - Hxy*v
    H = Hxx - Hxy*solve(Hyy,Hyx)
    H = (H+H^T)/2
    若 need_value: P_hat = f(x,y) - gy^T*v/2
    return P_hat（可省略）,g,H
```

RAW 不使用 Newton 修正梯度；CORRECTED 不重新查询已有 gy。各方法只调用主流程指定的构造器。

### 固定步数 Nesterov 内层

```text
AGD(x,y_in,N; ell,mu):
    kappa = ell/mu
    beta = (sqrt(kappa)-1)/(sqrt(kappa)+1)
    y = v = y_in                         # 每次调用重启动量
    for j=1,...,N:
        gv = grad_y f(x,v)               # 一次真实查询
        y_new = v + gv/ell
        v = y_new + beta*(y_new-y)
        y = y_new
    return y,N 次梯度查询                # N=0 时无查询，返回入口 y
```

固定步数分支不查询终点梯度，返回 grad_y=None、residual=None；converged=True 仅表示指定步数完成。
固定 N 的有效性来自外层误差预算。残差停止模式保持不变，仍测量并计入每次残差查询。

---

<a id="algorithm-6"></a>

## MCN（float64，残差上界初始化）

源码：algorithms/mcn.py；subproblem_solvers/cubic.py；inner_solvers/nesterov.py；scripts/run_comparision.py。

本文对应 **2026-09-24 工作区当前源码**（包括未提交改动），描述实际实现而非理想化论文算法。
`epsilon` 均为方法内部参数；原生停止不等于统一外部精度。代码块中的实数运算采用本文件标注的数值后端。
每次真实的向量 `grad_y` 查询计作一次内层梯度；读缓存不计费。矩阵分解/特征求解不计入该计数。
每个算法章节包含自己的主流程和所需子程序。

### 参数与运行脚本初始化

```text
kappa=ell/mu; M=4*sqrt(2)*kappa^3*rho
A=min(epsilon/(192*ell), sqrt(M*epsilon)/(48*rho))
stop_radius=sqrt(epsilon/M)/2
```

MCN 默认 K0=None，在正式 run 内执行以下初始化，时间和一次查询均计入算法成本：

```text
D0=||grad_y f(x0,y_minus1)||/mu
K0=0（D0=0）否则 max(0,ceil(2*sqrt(kappa)*log(sqrt(kappa+1)*D0/A)))
```

该界对任意 y_minus1 成立，无需高精度 preparation solve。每次 run 都重新校准，避免复用旧初始化的 K0。
仍支持显式传入已有证书的 K0；这种模式不另查入口梯度。
默认模式总计数为 1+sum(K_t)，history 的 inner_grad_evals 首项记录初始化查询。
未测量的终点 residual 记为 None；独立评价成本仍单列。


### MCN 主流程

```text
输入 x0,y_minus1,K0,epsilon,外层预算 Qmax
x=x0; y_prev=y_minus1
for t=0,...,Qmax-1:
    若 t=0: N=K0
    否则:
        N=ceil(2*sqrt(kappa)*log(sqrt(kappa+1)*(A+kappa*||s_prev||)/A))
    y=AGD(x,y_prev,N)
    (g,H)=RAW(x,y)
    s=CUBIC(g,H,M)
    若子问题未收敛: 抛出异常，由 runner 记录失败
    若 ||s||<=stop_radius:
        return 成功，x+s,y
    x=x+s; y_prev=y; s_prev=s
```

**当前源码事实：** MCN 类在耗尽 for 循环后没有显式 return，Python 返回 None；runner 将其记录为失败。
本伪代码没有把尚未实现的统一失败返回补写成已有功能。终点 y 仍属于更新前的 x，评价器会重新求内层。

### 当前三次子问题实现

目标 `g^T s+s^T H s/2+M*||s||³/6`，要求 M>0。

```text
CUBIC(g,H,M):
    检查有限输入、维度和 dtype；对称化 H
    [theta,V]=eigh(H); a=V^T*g
    lambda0=max(0,-theta_min)
    eta=max(tol,64*machine_epsilon)
    gaps=theta-theta_min（lambda0>0）否则 theta

    若 lambda0=0 且 ||g||=0:
        return VERIFY(s=0,lambda=0)

    若 lambda0>0:
        Imin={i: gaps_i<=eta*max(||H||_2,machine_tiny)}
        若 ||a[Imin]||<=eta*||g||:
            z0[Imin]=0；其余 z0_i=-a_i/gaps_i
            r0=||z0||；R=2*lambda0/M
            若 R 有限且正、r0<=R*(1+eta):
                alpha=R*sqrt(max(0,1-(r0/R)^2))
                z0[0]=alpha
                candidate=VERIFY(V*z0,lambda0)
                若 candidate.converged: return candidate

    # 普通根：用 delta=lambda-lambda0，避免 lambda0+小偏移的消去
    phi(delta)=||-a/(gaps+delta)||-2*(lambda0+delta)/M
    left=0
    right=max(lambda0,||H||_2,sqrt(||g||)*sqrt(M),machine_tiny)
    在 max_iter 次预算内倍增 right，直到 phi(right)<=0 且有限
    若未括住根或发生溢出: 报告失败
    二分 delta∈(left,right)，最多 max_iter 次:
        z=-a/(gaps+delta)
        若相对范数条件满足:
            candidate=VERIFY(V*z,lambda0+delta)
            若 candidate.converged: return candidate
        按 phi(delta) 的符号收缩区间
    返回最后候选及 VERIFY 的结果，不能无检查地标为成功
```

`VERIFY` 在原坐标检查有限性、lambda>=0，以及：

```text
r=||s||；h=||H||_2
||(H+lambda*I)s+g|| <= eta*max(||g||+(h+|lambda|)*r,machine_tiny)
max(0,-theta_min-lambda) <= eta*max(h+|lambda|,machine_tiny)
|lambda-M*r/2| <= eta*max(|lambda|,M*r/2,machine_tiny)
```

默认 tol=1e-14、max_iter=1000；tol 现在是相对容差。成功表示满足数值全局最优性条件。
覆盖非零 g 的一般 hard case、重最小特征值、边界半径相等，以及零投影但仍需普通求根的情况。
PSD 分支只对真正零梯度直接返回零步，避免把小非零梯度抹掉。
数值最小特征空间的候选也必须通过 VERIFY；不通过时继续普通求根。
当前保持 cubic/TR 的独立实现，没有改动 TR 求解器。

### 导数与值构造

记 `q_x(y)=-f(x,y)`，`gy=grad_y f(x,y)`，Hessian 分块为 `Hxx,Hxy,Hyx,Hyy`。

```text
RAW(x,y):
    g = grad_x f(x,y)
    H = Hxx - Hxy * solve(Hyy,Hyx)
    return g, (H+H^T)/2

CORRECTED(x,y,缓存 gy, need_value):
    v = solve(Hyy,gy)
    g = grad_x f(x,y) - Hxy*v
    H = Hxx - Hxy*solve(Hyy,Hyx)
    H = (H+H^T)/2
    若 need_value: P_hat = f(x,y) - gy^T*v/2
    return P_hat（可省略）,g,H
```

RAW 不使用 Newton 修正梯度；CORRECTED 不重新查询已有 gy。各方法只调用主流程指定的构造器。

### 固定步数 Nesterov 内层

```text
AGD(x,y_in,N; ell,mu):
    kappa = ell/mu
    beta = (sqrt(kappa)-1)/(sqrt(kappa)+1)
    y = v = y_in                         # 每次调用重启动量
    for j=1,...,N:
        gv = grad_y f(x,v)               # 一次真实查询
        y_new = v + gv/ell
        v = y_new + beta*(y_new-y)
        y = y_new
    return y,N 次梯度查询                # N=0 时无查询，返回入口 y
```

固定步数分支不查询终点梯度，返回 grad_y=None、residual=None；converged=True 仅表示指定步数完成。
固定 N 的有效性来自外层误差预算。残差停止模式保持不变，仍测量并计入每次残差查询。
