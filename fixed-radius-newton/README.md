# Fixed-radius parameter-free minimax method

`main.tex` 保留 fixed-radius trust-region 与 dual Newton tracking 的参数自由版本。
文稿包含问题假设、校正值/梯度/Hessian、初始化与主算法、全局复杂度结果，以及所需证明。

- 主文件：[`main.tex`](main.tex)
- 编译结果：[`main.pdf`](main.pdf)
- 编译：在仓库根目录运行 `make fixed-radius`

参数自由性指算法不输入正则性参数；主定理中的精度适用条件仍按文稿陈述。
这一 minimax 版本与并列的 `universal-sharp-utr/` 无约束精确 oracle 版本分别维护。
