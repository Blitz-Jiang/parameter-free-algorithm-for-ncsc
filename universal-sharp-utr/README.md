# Universal sharp second-order UTR

无约束、精确 oracle 的曲率增强 UTR 文稿。算法使用全局求解的二次信赖域子问题，
不输入或估计广义光滑性常数，梯度精度与曲率精度可以独立指定。

| 文件 | 内容 |
| --- | --- |
| [`main.tex`](main.tex) | 主入口与摘要 |
| [`preamble.tex`](preamble.tex) | 排版、符号及定理环境 |
| [`sections/problem.tex`](sections/problem.tex) | 问题设定、广义光滑性假设、驻点定义 |
| [`sections/algorithm.tex`](sections/algorithm.tex) | 算法、接受条件，以及引用的标准信赖域最优性条件 |
| [`sections/analysis.tex`](sections/analysis.tex) | 技术分析入口与显式对数项定义 |
| [`sections/local_steps.tex`](sections/local_steps.tex) | 非零步长、安全回溯、接受步记号，以及统一的边界步/内部步集合与估计 |
| [`sections/potential.tex`](sections/potential.tex) | 联合势函数、尺度求和，以及与精度无关的初始阶段，各结论后接证明 |
| [`sections/main_theorem.tex`](sections/main_theorem.tex) | 主定理：四项主复杂度加与精度无关的加性对数项 |
| [`sections/complexity.tex`](sections/complexity.tex) | 主证明的三步：分类求和、利用未满足的停机条件计数、统计拒绝试探及 oracle 查询 |
| [`references.bib`](references.bib) | 参考文献 |
| [`main.pdf`](main.pdf) | 完整编译文稿 |
| [`main.bbl`](main.bbl) | 生成的参考文献，供直接预览主文件时使用 |

在仓库根目录运行 `make second-order`。编译完成后自动清理临时文件，
仅保留本目录中的 `main.pdf` 和 `main.bbl`，不再创建 `out/` 中的重复产物。

算法统一使用每次尝试的索引 $k$ 与变量 $x_k,\sigma_k$；拒绝时保持点不变。
分析用 $\tau_j$ 标出第 $j$ 个接受试探在原算法中的索引，区分尝试次数与接受步数。
证明统一用 $\mathcal B,\mathcal I$ 表示边界步与内部步，
$g_j,\chi_j,\Delta_j$ 分别表示接受点的梯度范数、负曲率和函数下降量。
联合势函数先控制尺度，初始阶段引理再给出与精度无关的 $j_0$；
主证明只对 $j\ge j_0$ 的两类步求和，并用 Young 不等式直接推出四项复杂度。
所有内容均在正文中；除引用的标准最优性条件外，各引理、命题和主定理后直接紧接证明，不设附录。

本文的全局收敛性指：在所述假设下，从任意初始点出发，对任意正的梯度与曲率精度，
算法有限终止并满足指定的近似二阶驻点条件。主定理同时给出对每一组可行光滑性常数
均成立的复杂度上界；未断言整个迭代序列收敛到同一个点。
