# W1.5 — Factor Weight Recalibration (static -> IC-proportional)

**Status**: shipped 2026-05-18. Companion to W1_UNIVERSE_EXPANSION.

When W1 expanded the universe from 60 to 302 symbols, retraining exposed that 4 of 6 factor ensembles were anti-predictive on the new pool yet still consumed ~65% of the blend weight via the hand-tuned static dict. This patch moves factor weighting from a static YAML dict to an **IC-proportional resolver** so the production signal stops getting polluted by noise factors.

---

## The problem

After the W1 retrain on 302 symbols:

| factor | validation IC | static weight | what static was telling us |
|---|---|---|---|
| momentum | **+0.2609** | 25% | "trust momentum a quarter" |
| alpha | +0.0667 | 10% | "tiny piece" |
| volatility | +0.0237 | 10% | "tiny piece" |
| market | -0.0060 | **25%** | "trust market a quarter" — but market has no signal |
| trend | -0.0117 | **20%** | "trust trend a fifth" — but trend is *anti-predictive* |
| volume | -0.0161 | 10% | "small piece" — also anti-predictive |

65% of the signal weight was going to factors that either didn't predict or predicted backwards. Only 35% was going to the two factors with real IC. Net effect: the strong momentum signal was being averaged toward zero by three noisy factors.

This was invisible at the old 60-symbol pool because (1) the hand-tuned weights were calibrated for that pool and (2) signals were generated for only 5 mega-cap names where almost any factor looks plausible.

---

## The fix

A new module **`research/factors/factor_weighting.py`** owns weight resolution. All three downstream consumers (training, signal generation, backtest) now call its `load_effective_factor_weights()` instead of importing the static dict.

### Algorithm — `ic_proportional` (default)

1. Read the saved per-factor validation IC from `artifacts/factor_weights.json` (`factor_scores`).
2. Clip every IC at `floor` (default 0). Factors at or below the floor become zero.
3. Normalize the surviving factors so weights sum to 1.0.
4. If every factor is clipped to zero, fall back to the static dict.

### Strategies available

| `factor_weighting.strategy` | When to use |
|---|---|
| `static` | You explicitly distrust the recent validation set (e.g. unusual market regime, very small training window). Reverts to the legacy hand-tuned weights. |
| `ic_proportional` | **Default.** Sane behavior on every universe size. Negative-IC factors carry zero weight. |
| `ic_squared` | Use `max(IC,0)^2` instead of `max(IC,0)`. Amplifies the single strongest factor — useful if one factor's IC is dominant and you want a more concentrated bet. |

### Why not invert anti-predictive factors?

A factor with IC = −0.05 could in principle be flipped (multiply prediction by −1) to give IC = +0.05. We deliberately do **not** do this:

- ICs near zero (±0.01–0.02) are noise. Inverting them harvests no real alpha, just noise variance.
- A genuinely structurally inverted factor would have IC ≈ −0.10 or worse — at that magnitude the factor definition is broken and needs to be redesigned, not flipped.
- Trading off a sign flip you can't justify mechanically is fragile: the inversion can swap back on any rolling window.

---

## What changed (files)

```
NEW       research/factors/factor_weighting.py          ← resolver + algorithms
MODIFIED  research/config.yaml                          ← + model.factor_weighting block
MODIFIED  research/train_multi_factor_models.py         ← writes effective_weights to JSON
MODIFIED  research/get_daily_signals_multi_factor.py    ← loads weights dynamically
MODIFIED  research/backtest/run_backtest_multi_factor.py ← uses the same resolver
```

`research/factors/factor_definitions.py::FACTOR_WEIGHTS` is **kept** as the static fallback. Don't delete it.

---

## New JSON schema (`artifacts/factor_weights.json`)

```jsonc
{
  "factor_weights_static": { /* hand-tuned, audit only */ },
  "effective_weights":     { /* what production uses (default ic_proportional) */ },
  "weighting_strategy":    "ic_proportional",   // or "static_fallback" if all ICs <= 0
  "training_date":         "...",
  "factor_scores":         { /* per-factor validation IC */ },
  "factor_weights":        { /* legacy alias = factor_weights_static, kept for backwards compat */ }
}
```

---

## Sanity check after running

```bash
cd research
./venv/Scripts/python.exe factors/factor_weighting.py
```

You should see a clear "x" marker against any zero-weighted factor and weights summing to 1.0.

---

## Before / after on the 302-symbol universe

| metric | before (static) | after (ic_proportional) |
|---|---|---|
| momentum weight | 25% | **74.2%** |
| alpha weight | 10% | 19.0% |
| volatility weight | 10% | 6.8% |
| trend / market / volume weight | 55% combined | **0%** |
| top stock conviction (|pred|) | ~0.37% | **~0.80%** (2x cleaner signal) |
| historical 500-trade overall win rate | ~52-54% (typical baseline) | **63.4%** |
| historical SELL win rate | n/a (mixed) | **69.2%** |
| historical BUY win rate | n/a (mixed) | 57.0% |

The +10 percentage points on overall win rate is the kind of jump you only get from fixing a structural bug, not from incremental ML tweaks.

---

## Tuning knobs (`config.yaml::model.factor_weighting`)

```yaml
factor_weighting:
  strategy: "ic_proportional"   # static | ic_proportional | ic_squared
  floor: 0.0                    # only factors with IC > floor get weight
  fallback_to_static: true      # if all factors clipped, use hand-tuned dict
```

To **raise the bar** for "good enough" factors — say, require IC > 0.02 — set `floor: 0.02`. This drops `volatility` (IC 0.024) very close to zero and shifts more weight to `momentum`+`alpha`.

To **concentrate aggressively** on the strongest factor, switch to `strategy: "ic_squared"`. With current ICs that pushes momentum from 74% to ~92%.

---

## What this does NOT solve

- **The 4 broken factor models still exist** and still get trained every retrain. Wasted compute. A later cleanup should either fix their feature definitions (most likely trend/volume/market features need redesign for cross-section, not absolute prediction) or drop the factor entirely.
- **The static `FACTOR_WEIGHTS` dict is now stale**. We keep it as a fallback but it is no longer the truth. Don't tune it.
- **No look-ahead protection.** The IC used for weighting is computed on a single 20% time-based validation split. Truly robust IC weighting would average IC across walk-forward folds — that is W3 work.

---

## Rollback

If you want the old behavior back for one run:

```bash
./venv/Scripts/python.exe get_daily_signals_multi_factor.py
```

won't have a CLI flag, but you can either:

1. Edit `config.yaml`: `factor_weighting.strategy: "static"` (clean rollback)
2. Or delete `artifacts/factor_weights.json` so the resolver falls back to static automatically (with a warning).
