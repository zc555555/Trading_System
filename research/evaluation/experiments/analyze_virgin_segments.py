"""Segment-level verdict for the vol_ext adoption decision.

Splits the extended-history walk-forward panels into four evidence tiers by
how 'virgin' each period is for the vol_ext decision:

  virgin_early : test dates before 2021-12-30 -- never evaluated by ANY
                 experiment before the history extension (robustness tier:
                 requirement is 'not systematically negative', not 'strong')
  seen_dev     : 2021-12-30 .. 2025-06-30 -- the original dev window
                 (already looked at; informational only)
  holdout      : 2025-07-01 .. 2026-05-15 -- partially spent by the prior
                 decision discussion
  fresh        : after 2026-05-15 -- data fetched today; fully virgin and
                 the most decision-relevant tier

Compares baseline (tag 'ext') vs vol_ext variant (tag 'volC_ext') blended
IC per tier, plus vol_ext's standalone factor IC.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from evaluation.metrics import daily_rank_ic, summarize_ic  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
TARGET = "future_return_5d"
NW_LAGS = 4

SEGMENTS = [
    ("virgin_early", None, "2021-12-30"),
    ("seen_dev", "2021-12-30", "2025-07-01"),
    ("holdout", "2025-07-01", "2026-05-16"),
    ("fresh", "2026-05-16", None),
]


def _seg_mask(dates: pd.Series, start: str | None, end: str | None):
    tz = dates.dt.tz
    mask = pd.Series(True, index=dates.index)
    if start:
        mask &= dates >= pd.Timestamp(start).tz_localize(tz)
    if end:
        mask &= dates < pd.Timestamp(end).tz_localize(tz)
    return mask


def summarize_panel(panel: pd.DataFrame, col: str) -> dict:
    out = {}
    for name, start, end in SEGMENTS:
        seg = panel[_seg_mask(panel['date'], start, end)]
        out[name] = summarize_ic(daily_rank_ic(seg, col, target_col=TARGET),
                                 nw_lags=NW_LAGS)
    return out


def main():
    base = pd.read_parquet(RESULTS / "oos_predictions_h5_ext.parquet")
    volc = pd.read_parquet(RESULTS / "oos_predictions_h5_volC_ext.parquet")

    rows = []
    base_pred = summarize_panel(base, 'pred')
    volc_pred = summarize_panel(volc, 'pred')
    ext_factor = summarize_panel(volc, 'factor_vol_ext')

    print("=" * 86)
    print("VOL_EXT VERDICT BY EVIDENCE TIER  (h=5 blended rank IC, NW lags 4)")
    print("=" * 86)
    print(f"{'segment':<14}{'days':>6} | {'baseline':>18} | {'with vol_ext':>18} | "
          f"{'vol_ext alone':>18}")
    print("-" * 86)
    for name, _, _ in SEGMENTS:
        b, v, f = base_pred[name], volc_pred[name], ext_factor[name]

        def fmt(s):
            if not s['n_days']:
                return f"{'n/a':>18}"
            return f"{s['ic_mean']:>+8.4f} (t={s['t_stat']:>5.2f})"
        print(f"{name:<14}{b['n_days']:>6} | {fmt(b)} | {fmt(v)} | {fmt(f)}")
        rows.append({'segment': name, 'days': b['n_days'],
                     'baseline_ic': b['ic_mean'], 'baseline_t': b['t_stat'],
                     'volc_ic': v['ic_mean'], 'volc_t': v['t_stat'],
                     'ext_ic': f['ic_mean'], 'ext_t': f['t_stat']})

    pd.DataFrame(rows).to_csv(RESULTS / "vol_ext_verdict_segments.csv", index=False)
    print("-" * 86)
    print("decision guide: adopt if vol_ext is not systematically negative in "
          "virgin_early AND\nthe blend does not degrade there, AND fresh-tier "
          "evidence is favorable (small n -- weigh direction, not t).")
    print(f"saved: {RESULTS / 'vol_ext_verdict_segments.csv'}")


if __name__ == "__main__":
    main()
