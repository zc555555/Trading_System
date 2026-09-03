# Mutation-injection report
*2026-09-03T15:35:29.379007; 8 probes; false alarms on clean run: 0*

| mutation | real instance | detected | fired oracles |
|---|---|---|---|
| clean | - | no | - |
| feature_future_shift | dpo_20 shift(-11) (audit 2026-07) | yes | future_leak, implausible_strength, controls |
| centered_window | AQuA Appendix B full-day normaliser | yes | future_leak, controls |
| label_misaligned | label from close[t-1]: today's return inside the label | yes | implausible_strength, controls |
| pit_mask_off | today's members applied to history (PIT experiment 2026-08) | yes | membership_mask |
| holdout_exposed | selection leakage: a hidden tier reaches the agent | yes | visibility |
| dedupe_off_flip | direction calibration after seeing dev (AQuA step we forbid) | yes | audit_duplicate |
