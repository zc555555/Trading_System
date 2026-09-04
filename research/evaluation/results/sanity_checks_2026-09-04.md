# Sanity screens of new DSL fields (2026-09-04)

Screen-stage only, `--no-record`: none of these rows is in the mined ledger or
in any BH family (RULEBOOK "Fields (recorded 2026-09-04, mining paused)").
Textbook expressions, direction declared from the literature, seen_dev
2021-12-30 .. 2025-07-01, rank IC with Newey-West t.

## XBRL fundamentals, h = 20 (all FAIL the screen rule |t| >= 2)

| expression | dir | dev IC | dev t | coverage |
|---|---|---|---|---|
| rank(book_to_market) | + | +0.0064 | +0.37 | 0.98 |
| rank(gross_profitability) | + | -0.0051 | -0.39 | 0.49 |
| rank(asset_growth) | - | +0.0120 | +0.91 | 0.99 |
| rank(accruals) | - | -0.0049 | -0.42 | 0.99 |
| rank(earnings_yield) | + | +0.0187 | +1.17 | 0.97 |
| rank(roe) | + | +0.0121 | +0.93 | 0.92 |
| rank(cash_to_assets) | + | +0.0124 | +0.68 | 0.99 |
| rank(gp) - rank(asset_growth) - rank(accruals) | + | +0.0083 | +0.69 | 0.49 |

Value, profitability, investment and accrual anomalies are flat on S&P 500
large caps over 2022-2025 at a 20-session horizon; asset growth even has
the wrong sign. Coverage: gross profit only exists for the half of the
universe that reports COGS or GrossProfit.

## FINRA Reg SHO daily short-volume ratio, h = 5

| expression | dir | dev IC | dev t | coverage | note |
|---|---|---|---|---|---|
| ts_mean(fillna(short_vol_ratio, 0), 5) | - | -0.0184 | -3.41 | 1.00 | PASS; rho 0.32 with the prior short_ratio level, residual t -2.23 |
| ts_mean(fillna(short_vol_ratio, 0), 20) | - | -0.0199 | -3.40 | 1.00 | same cluster as the 5-day mean (rho 0.81) |
| 5-day mean - 60-day mean | - | -0.0042 | -1.04 | 1.00 | change version flat |
| ts_zscore(..., 60) | - | -0.0053 | -1.62 | 1.00 | flat |

The LEVEL of the daily short-volume share is the strongest single-field
seen_dev signal found at h = 5 in the whole programme (previous best: short
interest level, t -2.9); its change is not informative. Unseen tiers are
unknown by construction of this check. Data from 2017-12-29 only.

## SEC Form 13F institutional ownership, h = 20 (all FAIL the screen rule)

| expression | dir | dev IC | dev t | coverage |
|---|---|---|---|---|
| delta(inst_holders, 63) / (delay(inst_holders, 63) + 1) (breadth change) | + | +0.0167 | +0.93 | 0.98 |
| delta(inst_own, 63) | + | -0.0035 | -0.47 | 0.96 |
| rank(inst_own) (level) | + | -0.0267 | -2.75 | 0.96 |
| rank(inst_top5) | - | -0.0081 | -0.58 | 0.98 |

Breadth change has the literature sign but is weak; the ownership LEVEL is
significantly negative on seen_dev (high institutional ownership, lower
20-session returns), the opposite of the declared direction and most
likely a crowding / size proxy. Not pursued while rounds are paused.
