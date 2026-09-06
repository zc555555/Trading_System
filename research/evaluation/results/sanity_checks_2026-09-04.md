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

## Wikipedia page views (attention), h = 5 (all FAIL the screen rule)

| expression | dir | dev IC | dev t | coverage |
|---|---|---|---|---|
| log(views) - 60-day mean of log(views) (attention shock) | - | +0.0010 | +0.25 | 0.97 |
| 5-day mean of log(views) - 60-day mean | - | -0.0030 | -0.61 | 0.97 |
| rank(20-day mean of log(views)) (level) | - | +0.0080 | +1.16 | 0.97 |
| attention shock x sign(5-day return) | - | +0.0097 | +2.52 | 0.97 |

No attention-reversal effect on S&P 500 large caps at a 5-session horizon.
The interaction is significant with the OPPOSITE sign to the declared one
(attention shocks accompanying a move are followed by continuation, not
reversal); it fails the screen because the direction was pre-declared, and
it is not pursued while rounds are paused.

## Coverage of the new fields on seen_dev (both caches, 36 columns)

inst_own 0.94, inst_holders 0.96, inst_top5 0.96, wiki_views 0.96,
short_vol_ratio 0.96; fundamentals 0.89-0.97 except gross_profitability
0.48, op_margin 0.69, capex_to_assets 0.80.

## Mid-cap universe (2026-09-06), h = 20, textbook factors (not recorded)

| expression | dir | dev IC | dev t | coverage | note |
|---|---|---|---|---|---|
| delay(ts_sum(returns, 229), 20) (12-1 momentum) | + | +0.039 | +2.15 | 0.98 | PASS; rho 0.73 with production pct_52w_high (rewrite), residual t 3.0 |
| ts_sum(returns, 20) (short-term reversal) | - | +0.005 | +0.40 | 1.00 | flat |
| log(marketcap) (size within mid caps) | - | +0.008 | +0.78 | 0.97 | flat |
| ts_std(returns, 60) (low vol) | - | -0.039 | -1.79 | 1.00 | identical to production volatility_60d |
| ts_mean(turnover, 60) | - | -0.006 | -0.46 | 0.96 | flat |
| ts_mean(fillna(short_vol_ratio, 0), 20) | - | -0.011 | -1.52 | 0.97 | weaker than on the S&P (t -3.4) |
| delay(ts_sum(returns, 20), 228) (seasonality) | + | +0.035 | +3.27 | 0.98 | PASS, representative |
| sector_rank(12-1 momentum) | + | +0.030 | +1.94 | 0.98 | just below the bar |

Same development segment (2021-12-30 .. 2025-07-01), ~920 point-in-time
names per date. Momentum-type signals carry IC 0.03-0.04 here versus 0.01-0.02
on the S&P 500; reversal, size and turnover are flat in this period.
