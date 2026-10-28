# Universal sharp second-order UTR

无约束、精确 oracle 的曲率增强 UTR 文稿。算法使用全局求解的二次信赖域子问题，
不输入或估计广义光滑性常数，梯度精度与曲率精度可以独立指定。

| 文件 | 内容 |
| --- | --- |
| [`main.tex`](main.tex) | 主入口与摘要 |
| [`preamble.tex`](preamble.tex) | 排版、符号及定理环境 |
| [`sections/problem.tex`](sections/problem.tex) | 问题设定、广义光滑性假设、驻点定义 |
| [`sections/algorithm.tex`](sections/algorithm.tex) | 算法、接受条件，以及信赖域最优性引理及其紧随的证明 |
| [`sections/analysis.tex`](sections/analysis.tex) | 技术分析入口与显式对数项定义 |
| [`sections/local_steps.tex`](sections/local_steps.tex) | 非零步长、安全回溯、接受索引和边界/内点分析，各引理后接证明 |
| [`sections/potential.tex`](sections/potential.tex) | Young 不等式与联合势函数，各引理、命题后接完整证明 |
| [`sections/main_theorem.tex`](sections/main_theorem.tex) | 主定理及紧随的证明，包含零初始间隙情形 |
| [`sections/complexity.tex`](sections/complexity.tex) | 主定理证明中的梯度初始阶段、求和、有限终止与 oracle 计数 |
| [`references.bib`](references.bib) | 参考文献 |
| [`main.pdf`](main.pdf) | 完整编译文稿 |

在仓库根目录运行 `make second-order`。编译缓存写入 `out/`。

算法统一使用每次尝试的索引 $k$ 与变量 $x_k,\sigma_k$；拒绝时保持点不变。
分析用 $\tau_j$ 标出第 $j$ 个接受试探在原算法中的索引，区分尝试次数与接受步数。
所有内容均在正文中；各引理、命题和主定理后直接紧接证明，不设附录。

本文的全局收敛性指：在所述假设下，从任意初始点出发，对任意正的梯度与曲率精度，
算法有限终止并满足指定的近似二阶驻点条件。主定理同时给出对每一组可行光滑性常数
均成立的复杂度上界；未断言整个迭代序列收敛到同一个点。
