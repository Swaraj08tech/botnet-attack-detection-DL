# Notebook 07: the result in simple words

| Inputs | MLP mean macro F1, three seeds | Saved LightGBM macro F1 |
|---|---:|---:|
| original42 | 0.5198 | 0.6307 |
| total_and_byte44 | 0.5241 | 0.6320 |

The MLP is a useful **bounded neural reference**, not our best-performing model. All six MLP runs correctly classify **0 of 13 Worms** rows. The matching LightGBM models find 8/13 and 9/13.

Adding total bytes and byte balance raises mean MLP macro F1 by 0.44 percentage points. Paired seed changes are +1.73, -1.20 and +0.78 points: some runs improve and another worsens. This is not a stable improvement across seeds or proof of performance on new traffic.

For 44 features and fixed seed 42, the MLP finds **271/1,226 DoS** rows, versus **86/1,226** for LightGBM. But it finds fewer Analysis (21 vs 30), Backdoor (15 vs 26), Exploits (2,942 vs 3,192), Fuzzers (1,304 vs 1,391) and Worms (0 vs 9). The full notebook keeps all per-class counts, false alarms, costs and seed variation visible.

We tried just **two configurations for 30 epochs**, using three grouped folds inside fitting. Falling training loss means fitting is progressing; it proves neither generalization nor convergence. No settings, epochs, seeds or weighting were changed in this review. Stronger MLP optimization would need a separately versioned experiment with its search and stopping rules declared first and selected only in fitting-only grouped folds.

The existing rule keeps total_and_byte44 as the provisional MLP input and seed 42 as the reference checkpoint. Keep both input sets and the exact LightGBM controls for Notebook 08. Model-selection has prior exposure; these remain optimistic development results. Calibration labels and official-test records are reserved. This is flow intrusion classification, with no DNS/temporal/botnet-specific or final-test claim.
