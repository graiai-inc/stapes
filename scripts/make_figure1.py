"""Generate Figure 1 for the stapes paper: best on-device vs best cloud API
on each of three datasets, across both WER (lower-is-better) and CTR
(higher-is-better).

Reads: paper/table1_wer.csv, paper/table2_ctr.csv
Writes: paper/figure1.png (300 dpi RGB), paper/figure1.pdf (vector).

npj Digital Medicine figure guidelines followed:
  * 300 dpi, RGB
  * Sans-serif (Helvetica-metric: Liberation Sans) at 8 pt print size
  * Colorblind-safe palette (Wong / Okabe-Ito blue + vermillion)
  * Panels labeled lowercase bold ``a``, ``b``
  * No truncated histogram axes
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent / 'paper'
WER_CSV = PAPER_DIR / 'table1_wer.csv'
CTR_CSV = PAPER_DIR / 'table2_ctr.csv'
OUT_PNG = PAPER_DIR / 'figure1.png'
OUT_PDF = PAPER_DIR / 'figure1.pdf'

# Colorblind-safe palette (Wong, Nat Methods 2011).
BLUE = '#0072B2'        # best on-device
VERMILLION = '#D55E00'  # best cloud

DATASET_COLS = [
    ('OSCE Std (n=272)*', 'OSCE (n=272)', 'OSCE\n(272 files)'),
    ('PriMock57 Std (n=57)', 'PriMock57 (n=57)', 'PriMock57\n(57 files)'),
    ('Psychiatric Std (n=71)', 'Psychiatric (n=71)', 'Psychiatric\n(71 files)'),
]


def _strip_dagger(val):
    if isinstance(val, str):
        s = val.replace('†', '').replace('‡', '').strip()
        # CTR cells look like "93.1 [92.8, 93.5]"; take the leading point estimate.
        s = s.split('[')[0].strip()
        if not s or s.lower() == 'n/a':
            return float('nan')
        return float(s.split()[0])
    return float(val)


def _best(df: pd.DataFrame, col: str, kind: str, lower_is_better: bool) -> tuple[float, str]:
    """Return (best value, model name) for the requested Type ('On-device' or
    cloud). Daggered (subset-only) rows are excluded from the best-of selection;
    all models are now evaluated on the full datasets, so none are excluded."""
    sub = df[df['Type'].str.startswith(kind)].copy()

    def is_subset(value) -> bool:
        return isinstance(value, str) and '†' in value

    sub = sub[~sub[col].apply(is_subset)]
    sub['_val'] = sub[col].apply(_strip_dagger)
    idx = sub['_val'].idxmin() if lower_is_better else sub['_val'].idxmax()
    return float(sub.loc[idx, '_val']), str(sub.loc[idx, 'Model'])


def main() -> None:
    wer = pd.read_csv(WER_CSV)
    ctr = pd.read_csv(CTR_CSV)

    # Collect values.
    datasets = [label for _, _, label in DATASET_COLS]
    wer_on = []
    wer_cloud = []
    for wer_col, _, _ in DATASET_COLS:
        best_on, _ = _best(wer, wer_col, 'On-device', lower_is_better=True)
        best_cloud, _ = _best(wer, wer_col, 'Cloud', lower_is_better=True)
        wer_on.append(best_on)
        wer_cloud.append(best_cloud)

    ctr_on = []
    ctr_cloud = []
    for _, ctr_col, _ in DATASET_COLS:
        best_on, _ = _best(ctr, ctr_col, 'On-device', lower_is_better=False)
        best_cloud, _ = _best(ctr, ctr_col, 'Cloud', lower_is_better=False)
        ctr_on.append(best_on)
        ctr_cloud.append(best_cloud)

    # Global rc setup for npj compliance.
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Liberation Sans', 'Nimbus Sans', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'pdf.fonttype': 42,   # TrueType-embeddable
        'ps.fonttype': 42,
    })

    # npj: 180 mm max width for 2-column. 7.09 inches at ~90 mm per panel.
    fig, (ax_wer, ax_ctr) = plt.subplots(
        1, 2, figsize=(7.09, 3.2), dpi=300,
    )

    x = np.arange(len(datasets))
    width = 0.36

    # Panel a - WER
    bars_wer_on = ax_wer.bar(
        x - width / 2, wer_on, width, label='Best on-device',
        color=BLUE, edgecolor='black', linewidth=0.5,
    )
    bars_wer_cloud = ax_wer.bar(
        x + width / 2, wer_cloud, width, label='Best cloud API',
        color=VERMILLION, edgecolor='black', linewidth=0.5,
    )
    ax_wer.set_ylabel('Word error rate (%)')
    ax_wer.set_title('WER (lower is better)', pad=6)
    ax_wer.set_xticks(x)
    ax_wer.set_xticklabels(datasets)
    ax_wer.set_ylim(0, max(wer_on + wer_cloud) * 1.22)
    ax_wer.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
    ax_wer.set_axisbelow(True)

    for bar, val in zip(bars_wer_on, wer_on):
        ax_wer.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35,
            f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold',
        )
    for bar, val in zip(bars_wer_cloud, wer_cloud):
        ax_wer.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35,
            f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold',
        )

    # Panel b - CTR
    ctr_min = min(ctr_on + ctr_cloud)
    ctr_max = max(ctr_on + ctr_cloud)
    # Start y-axis at 0 to respect "no truncated histogram axes"; annotate
    # values to keep the ordering legible.
    bars_ctr_on = ax_ctr.bar(
        x - width / 2, ctr_on, width, label='Best on-device',
        color=BLUE, edgecolor='black', linewidth=0.5,
    )
    bars_ctr_cloud = ax_ctr.bar(
        x + width / 2, ctr_cloud, width, label='Best cloud API',
        color=VERMILLION, edgecolor='black', linewidth=0.5,
    )
    ax_ctr.set_ylabel('Clinical term recall (%)')
    ax_ctr.set_title('CTR (higher is better)', pad=6)
    ax_ctr.set_xticks(x)
    ax_ctr.set_xticklabels(datasets)
    ax_ctr.set_ylim(0, 110)
    ax_ctr.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
    ax_ctr.set_axisbelow(True)

    for bar, val in zip(bars_ctr_on, ctr_on):
        ax_ctr.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.4,
            f'{val:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold',
        )
    for bar, val in zip(bars_ctr_cloud, ctr_cloud):
        ax_ctr.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.4,
            f'{val:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold',
        )

    # Panel labels.
    for ax, label in [(ax_wer, 'a'), (ax_ctr, 'b')]:
        ax.text(
            -0.14, 1.04, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom', ha='left',
        )

    # Single shared legend above both panels. Per-panel legends collided with
    # the (tall) clinical-term-recall bars in panel b, hiding the psychiatric
    # value labels; a figure-level legend keeps both panels uncluttered.
    handles, labels = ax_wer.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='upper center', ncol=2, frameon=True,
        framealpha=1.0, fontsize=8, bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_PDF, bbox_inches='tight')
    plt.close(fig)

    print(f'Wrote {OUT_PNG}', flush=True)
    print(f'Wrote {OUT_PDF}', flush=True)
    print('Values (WER on-device / cloud):', flush=True)
    for name, on, cl in zip(datasets, wer_on, wer_cloud):
        print(f'  {name.replace(chr(10), " ")}: on={on:.2f}  cloud={cl:.2f}', flush=True)
    print('Values (CTR on-device / cloud):', flush=True)
    for name, on, cl in zip(datasets, ctr_on, ctr_cloud):
        print(f'  {name.replace(chr(10), " ")}: on={on:.1f}  cloud={cl:.1f}', flush=True)


if __name__ == '__main__':
    main()
