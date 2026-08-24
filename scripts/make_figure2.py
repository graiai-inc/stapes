"""Generate Figure 2 for the stapes paper: ROVER fusion effect by dataset.

For each dataset, plots three WER anchors:
  * Best single on-device model
  * Best ROVER fusion pair
  * Best cloud API (reference dashed line)

Reads:
  * paper/table1_wer.csv
  * paper/supplementary_table_S2_rover_pairs.csv
  * results/figshare_fix_summary.tsv (confirms OSCE pair WER of 11.01 %)

Writes: paper/figure2.png (400 dpi RGB, clears Snapp's 1500x1200 px minimum), paper/figure2.pdf (vector).

npj Digital Medicine compliance:
  * 400 dpi, RGB
  * Sans-serif 8 pt (Liberation Sans - Helvetica-metric)
  * Colorblind-safe palette (Wong 2011)
  * No truncated y-axis (starts at 0)
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PAPER_DIR = REPO_ROOT / 'paper'
RESULTS_DIR = REPO_ROOT / 'results'
WER_CSV = PAPER_DIR / 'table1_wer.csv'
PAIRS_CSV = PAPER_DIR / 'supplementary_table_S2_rover_pairs.csv'
# OSCE pair WERs with the corrected (apostrophe-repaired) references live in
# figshare_fix_summary.tsv; supplementary_table_S2_rover_pairs.csv only
# contains PriMock57 + psychiatric pairs.
OSCE_PAIRS_TSV = RESULTS_DIR / 'figshare_fix_summary.tsv'
EXPORT_DPI = 400  # 300 dpi left both figures under 1200 px tall
OUT_PNG = PAPER_DIR / 'figure2.png'
OUT_PDF = PAPER_DIR / 'figure2.pdf'

# Wong (Nat Methods 2011) palette: colorblind-safe.
BLUE = '#0072B2'         # best single on-device
GREEN = '#009E73'        # best ROVER pair
VERMILLION = '#D55E00'   # best cloud reference

DATASET_LAYOUT = [
    # (display label, table1 column, supp S2 key or 'osce' for OSCE_PAIRS_TSV)
    ('OSCE', 'OSCE Std (n=272)*', 'osce'),
    ('PriMock57', 'PriMock57 Std (n=57)', 'primock57'),
    ('Psychiatric', 'Psychiatric Std (n=71)', 'nazmulkazi'),
]


def _strip_dagger(val) -> float:
    if isinstance(val, str):
        return float(val.replace('†', '').replace('‡', '').strip())
    return float(val)


def _best_by_type(df: pd.DataFrame, col: str, kind: str, lower_is_better: bool) -> tuple[float, str]:
    # Daggered (subset-only) rows are excluded; all models are now evaluated on
    # the full datasets, so none are currently excluded.
    sub = df[df['Type'].str.startswith(kind)].copy()

    def is_subset(value) -> bool:
        return isinstance(value, str) and '†' in value

    sub = sub[~sub[col].apply(is_subset)]
    sub['_val'] = sub[col].apply(_strip_dagger)
    idx = sub['_val'].idxmin() if lower_is_better else sub['_val'].idxmax()
    return float(sub.loc[idx, '_val']), str(sub.loc[idx, 'Model'])


def _best_pair(pairs: pd.DataFrame, osce_pairs: pd.DataFrame, dataset_key: str) -> tuple[float, str]:
    """Return (best fused WER %, pair name) for ``dataset_key``.

    PriMock57 and psychiatric use supplementary_table_S2_rover_pairs.csv.
    OSCE uses figshare_fix_summary.tsv (``fixed_wer`` column) because that is
    the post-correction recomputation of ROVER pair WERs on the repaired
    figshare-osce references.
    """
    if dataset_key == 'osce':
        sub = osce_pairs[osce_pairs['kind'] == 'pair'].copy()
        idx = sub['fixed_wer'].idxmin()
        return float(sub.loc[idx, 'fixed_wer']), str(sub.loc[idx, 'key'])

    sub = pairs[pairs['dataset'] == dataset_key].copy()
    idx = sub['wer_pct'].idxmin()
    return float(sub.loc[idx, 'wer_pct']), str(sub.loc[idx, 'models'])


def main() -> None:
    wer = pd.read_csv(WER_CSV)
    pairs = pd.read_csv(PAIRS_CSV)
    osce_pairs = pd.read_csv(OSCE_PAIRS_TSV, sep='\t')

    labels = []
    best_single = []
    single_models = []
    best_fused = []
    fused_models = []
    best_cloud = []
    cloud_models = []

    for label, wer_col, pair_key in DATASET_LAYOUT:
        on_val, on_name = _best_by_type(wer, wer_col, 'On-device', True)
        cl_val, cl_name = _best_by_type(wer, wer_col, 'Cloud', True)
        pair_val, pair_name = _best_pair(pairs, osce_pairs, pair_key)
        labels.append(label)
        best_single.append(on_val)
        single_models.append(on_name)
        best_fused.append(pair_val)
        fused_models.append(pair_name)
        best_cloud.append(cl_val)
        cloud_models.append(cl_name)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Liberation Sans', 'Nimbus Sans', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    # 7 inch wide figure, tall enough to show values cleanly.
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=300)

    x = np.arange(len(labels))
    group_half_width = 0.30  # horizontal offset for single/fused markers

    # Draw a connector between the best-single and best-fused values, then the
    # markers on top so the improvement from fusion is visible. Also draw a
    # short horizontal dashed line for the best cloud API, centred on the
    # dataset tick.
    for i, (single_val, fused_val, cloud_val) in enumerate(
            zip(best_single, best_fused, best_cloud)):
        xs = i - group_half_width
        xf = i + group_half_width
        # vertical connector between single and fused
        ax.plot(
            [xs, xf], [single_val, fused_val],
            color='#555555', linewidth=1.0, linestyle='-', zorder=1,
        )
        # cloud reference: short horizontal dashed line
        ax.plot(
            [i - 0.40, i + 0.40], [cloud_val, cloud_val],
            color=VERMILLION, linewidth=1.4, linestyle=(0, (4, 2)), zorder=2,
        )

    # Legend proxies first (using add only; actual markers placed next).
    ax.scatter(
        [x_i - group_half_width for x_i in x], best_single,
        s=90, color=BLUE, edgecolor='black', linewidth=0.6, zorder=3,
        label='Best single on-device model',
    )
    ax.scatter(
        [x_i + group_half_width for x_i in x], best_fused,
        s=90, color=GREEN, edgecolor='black', linewidth=0.6, marker='s', zorder=3,
        label='Best ROVER fusion pair',
    )
    # Single proxy handle for the cloud reference dashed line.
    ax.plot(
        [], [], color=VERMILLION, linewidth=1.4, linestyle=(0, (4, 2)),
        label='Best cloud API',
    )

    # Value labels.
    for i, val in enumerate(best_single):
        ax.annotate(
            f'{val:.2f}%',
            xy=(i - group_half_width, val),
            xytext=(-10, 0), textcoords='offset points',
            ha='right', va='center', fontsize=7.5, color=BLUE, fontweight='bold',
        )
    for i, val in enumerate(best_fused):
        ax.annotate(
            f'{val:.2f}%',
            xy=(i + group_half_width, val),
            xytext=(10, 0), textcoords='offset points',
            ha='left', va='center', fontsize=7.5, color=GREEN, fontweight='bold',
        )
    for i, val in enumerate(best_cloud):
        ax.annotate(
            f'{val:.2f}%',
            xy=(i + 0.40, val),
            xytext=(4, 0), textcoords='offset points',
            ha='left', va='center', fontsize=7.5, color=VERMILLION, fontweight='bold',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Word error rate (%)')
    ax.set_title('ROVER fusion effect by dataset')
    ymax = max(best_single + best_fused + best_cloud) * 1.30
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.65, len(labels) - 1 + 0.7)
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=True, framealpha=1.0)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=EXPORT_DPI, bbox_inches='tight')
    fig.savefig(OUT_PDF, bbox_inches='tight')
    plt.close(fig)

    print(f'Wrote {OUT_PNG}', flush=True)
    print(f'Wrote {OUT_PDF}', flush=True)
    for lab, s_val, s_mod, f_val, f_mod, c_val, c_mod in zip(
            labels, best_single, single_models, best_fused, fused_models,
            best_cloud, cloud_models):
        print(
            f'{lab:<12} single={s_val:.2f} ({s_mod})  fused={f_val:.2f} ({f_mod})  '
            f'cloud={c_val:.2f} ({c_mod})',
            flush=True,
        )


if __name__ == '__main__':
    main()
