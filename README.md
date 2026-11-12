# UTR manuscripts

本仓库保留三个并列的文稿目录。

| 目录 | 内容 | 阅读入口 |
| --- | --- | --- |
| [`fixed-radius-newton/`](fixed-radius-newton/) | Fixed-radius parameter-free minimax 版本，包含 dual Newton tracking、算法、主定理与证明 | [`main.tex`](fixed-radius-newton/main.tex) / [`main.pdf`](fixed-radius-newton/main.pdf) |
| [`jmlr-final/`](jmlr-final/) | JMLR 原稿及自包含修订稿，保留参考文献、实验章节和图件 | 原稿 [`main.tex`](jmlr-final/main.tex)；修订稿 [`revised_main.tex`](jmlr-final/revised_main.tex) |
| [`universal-sharp-utr/`](universal-sharp-utr/) | 无约束二阶 UTR：曲率增强、参数自由、universal sharp 复杂度 | [`main.tex`](universal-sharp-utr/main.tex) / [`main.pdf`](universal-sharp-utr/main.pdf) |

## 二阶版本的内容

`universal-sharp-utr/` 面向精确 oracle 和全局求解的二次信赖域子问题，包含：

- 广义光滑性假设与独立的梯度、曲率精度；
- 曲率增强 UTR 算法及接受/拒绝规则；
- 任意指定精度下的有限终止与 universal sharp 主定理；
- 安全回溯、联合势函数、驻点计数和 oracle 复杂度的完整证明。

## 编译

需要包含 `latexmk`、`pdflatex` 和 `bibtex` 的 TeX 环境。在仓库根目录运行：

```sh
make                 # 编译全部四个入口
make fixed-radius    # fixed-radius parameter-free
make jmlr-original   # JMLR 原稿
make jmlr-revised    # JMLR 修订稿
make second-order    # universal sharp 二阶 UTR
make clean           # 清理编译缓存，保留各目录中的最终 PDF
```

最终 PDF 位于对应的 TeX 入口旁边。`universal-sharp-utr/` 编译后自动清理临时文件，
只保留一套 `main.pdf` 和 `main.bbl`；其余文稿的编译缓存仍写入各自的 `out/`。
也可在 Codex 中打开文稿目录并选择相应主文件进行预览。

JMLR 修订稿使用的实验章节和静态图件位于其目录内，不依赖旧实验代码目录。
