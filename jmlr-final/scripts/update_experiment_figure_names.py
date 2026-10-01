#!/usr/bin/env python3
"""Paper-only legend updates from frozen data; never execute optimization.

Requires matplotlib, numpy, PyMuPDF. Run with --mode plots / --mode vector
when those packages are in separate Python environments. The latter accepts
--font-file with the full DejaVuSans.ttf font from matplotlib. Historical
reports and saved method IDs are read-only. Curves and axis limits are retained.
"""
from pathlib import Path
import argparse
import json
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'jmlr-final/figures/experiments'
DATA = ROOT / 'numerical_experiment/reports/abc_paper_assets'
LABELS = {'UTR5': 'Minimax UTR', 'UTR5-early': 'Minimax UTR-early'}


def plots():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    read = lambda name: json.loads((DATA / name).read_text())
    save = lambda fig, name: fig.savefig(OUT / (name + '.pdf'))
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    # Same panels / samples / styles as build_abc_paper_materials.py.
    mech = read('B_mechanism.json')
    paths = read('B_outer_trajectories.json')
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    for key, label in [('raw_error', 'Raw gradient'), ('corrected_error', 'Corrected gradient'), ('hessian_error', 'Schur Hessian')]:
        axs[0].loglog([p['delta'] for p in mech['perturbations']],
                      [p[key] for p in mech['perturbations']], '-o', ms=3,
                      label=f"{label}, slope {mech['slopes'][key]:.3f}")
    axs[0].set(xlabel='Inner perturbation magnitude', ylabel='Oracle error norm')
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.2)
    for key, label, style in [('exact_mcn', 'Exact MCN (theory M)', '--'),
                              ('exact_utr', 'Exact UTR', '-'),
                              ('early', LABELS['UTR5-early'], ':')]:
        run = paths[key]; ps = [p for p in run['history'] if p['K'] <= run['K_hit']]
        axs[1].plot([p['x'][0] for p in ps], [p['x'][1] for p in ps], style, lw=1.5, label=label)
    axs[1].set(xlabel='u', ylabel='v', title='Accepted outer path (through first hit)')
    axs[1].legend(fontsize=8); axs[1].grid(alpha=.2)
    fig.tight_layout(); save(fig, 'B_mechanism_and_path'); plt.close(fig)
    # Same exported scalar coordinates / truncation as plot_rbf_best_three.py.
    data = read('C_best_three_plot_data.json'); epsilon = 1e-6
    plt.rcParams.update({'font.size': 11})
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    for record in data:
        xx, ps, color = record['x'], record['points'], record['color']
        label = record['label'].replace('UTR5-early', LABELS['UTR5-early'])
        hit = next((i for i, p in enumerate(ps) if p['grad_norm'] <= epsilon), None)
        for ax, key in zip(axs, ['grad_norm', 'value', 'lambda_min']):
            yy = [p[key] for p in ps]
            ax.plot(xx, yy, color=color, lw=2 if record is data[0] else 1.5, label=label)
            if hit is not None: ax.scatter(xx[hit], yy[hit], s=30, color=color, zorder=5)
            ax.scatter(xx[-1], yy[-1], s=23, marker='s', color=color, zorder=4)
    for ax, label in zip(axs, [r'$\|\nabla P(x)\|$', r'$P(x)$', r'$\lambda_{\min}(\nabla^2 P(x))$']):
        ax.set_xscale('symlog', linthresh=10); ax.set_xlim(left=0); ax.grid(alpha=.18)
        ax.set_xlabel('Inner queries (initialization excluded)'); ax.set_ylabel(label)
    axs[0].set_yscale('log'); axs[0].axhline(epsilon, ls=':', color='.5', lw=1)
    axs[2].axhline(0, color='.35', lw=.8)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False)
    fig.text(.5, .015, 'Circle: first true-gradient hit; square: plotted endpoint. Numerical tails truncated; substantive rebounds retained.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .88)); save(fig, 'C_best_three'); plt.close(fig)


def vector_legends(font_file):
    import pymupdf as fitz
    if font_file is None:
        from matplotlib.font_manager import findfont
        font_file = findfont('DejaVu Sans')
    font = fitz.Font(fontfile=str(font_file))
    methods = ['MCN', 'GRTR', 'HSDA', 'UTR5', 'UTR5-early']
    colors = ['#D55E00', '#009E73', '#CC79A7', '#777777', '#0072B2']
    color = lambda h: tuple(int(h[i:i+2], 16)/255 for i in (1, 3, 5))
    # Stationarity is a composed PDF without a saved plotting generator. Rebuild
    # only its legend; all plot paths, axes and clipping rectangles stay intact.
    # The unused full-trajectory figure is handled the same way for consistency.
    with zipfile.ZipFile(ROOT / 'numerical_experiments_insert_package.zip') as z:
        for name, size, baseline, blank in [
            ('A_stationarity', 12, 40.31726, (55, 25, 550, 46)),
            ('A_all_trajectories', 10, 16.59766, (335, 2, 740, 23)),
        ]:
            doc = fitz.open(stream=z.read('numerical_experiments_ready/figures/experiments/' + name + '.pdf'), filetype='pdf')
            page = doc[0]
            page.add_redact_annot(fitz.Rect(blank), fill=(1, 1, 1))
            page.apply_redactions(images=0, graphics=0)
            labels = [LABELS.get(m, m) for m in methods]
            handle, pad, gap = size * 2, size * .8, size * 1.8
            widths = [handle + pad + font.text_length(s, fontsize=size) for s in labels]
            cursor = (page.rect.width - sum(widths) - 4*gap)/2
            page.insert_font(fontname='LegendSans', fontfile=str(font_file))
            for i, (text, width) in enumerate(zip(labels, widths)):
                y = baseline - .35*size
                page.draw_line((cursor, y), (cursor+handle, y), color=color(colors[i]),
                               width=1.5, dashes=None if i < 3 else '[5.5 2.4] 0' if i == 3 else '[1.5 2.4] 0')
                page.insert_text((cursor+handle+pad, baseline), text, fontname='LegendSans', fontsize=size)
                cursor += width + gap
            doc.subset_fonts(); doc.save(OUT / (name+'.pdf'), garbage=4, deflate=True); doc.close()
        # Archived extra panel figure: preserve every curve and axis. Its common
        # legend has a fixed-width column; modestly reduce this one label's font.
        name = 'C_comparison_detailed'
        doc = fitz.open(stream=z.read('numerical_experiments_ready/figures/experiments/'+name+'.pdf'), filetype='pdf')
        page = doc[0]; rect = page.search_for('UTR5-early')[0]
        page.add_redact_annot(rect, fill=(1, 1, 1)); page.apply_redactions(images=0, graphics=0)
        page.insert_font(fontname='LegendSans', fontfile=str(font_file))
        page.insert_text((rect.x0, 18.2593994140625), LABELS['UTR5-early'], fontsize=9, fontname='LegendSans')
        doc.subset_fonts(); doc.save(OUT/(name+'.pdf'), garbage=4, deflate=True); doc.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['all', 'plots', 'vector'], default='all')
    parser.add_argument('--font-file', type=Path)
    args = parser.parse_args()
    if args.mode in ['all', 'plots']: plots()
    if args.mode in ['all', 'vector']: vector_legends(args.font_file)


if __name__ == '__main__':
    main()
