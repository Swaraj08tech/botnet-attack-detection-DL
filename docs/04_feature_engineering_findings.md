# Notebook 04 — Feature Engineering

## What we created

We added three columns to the original 42 feature columns:

| New column | Formula | Meaning |
|---|---|---|
| fe_total_bytes | sbytes + dbytes | Total bytes in both directions |
| fe_byte_balance | (sbytes - dbytes) / (sbytes + dbytes) | Which direction carried more bytes |
| fe_packet_balance | (spkts - dpkts) / (spkts + dpkts) | Which direction carried more packets |

For either balance, +1 means entirely forward, -1 entirely back, and 0 equal amounts. We store 0 if the total is zero and keep the original counts to tell no traffic from equal traffic. For example, 900 bytes forward and 100 back give 1,000 total bytes and a byte balance of 0.8. Six packets forward and four back give a packet balance of 0.2.

The notebook shows these calculations directly in its code. The same formulas are also saved in src/data/feature_engineering.py and checked against the notebook calculations.

## What the training data shows

We studied only the 140,273 fitting rows, with 80,848 different original feature patterns. Repeated patterns are not proof of independent network events. The middle total-byte value is 802 bytes; the largest is 14,562,299. Neither balance has a zero total in these fitting rows.

The two balances are closely related: Spearman correlation is 0.9734. We keep both as candidates, but the next notebook should check whether both are worth using. Label summaries describe differences; labels never enter a formula, and these differences do not prove better detection.

We also checked packet rate and average bytes per packet. We left them out: average sizes closely match smean and dmean, while rate already describes similar information and its exact calculation remains uncertain. Full numbers and reasons are in results/tables/04_feature_decisions.csv and 04_redundancy_checks_fit.csv. No original feature was removed.

## What we saved and checked

Each saved row set now has 45 raw feature columns: train (140,273 rows), model_selection (17,534) and calibration (17,534). Inputs are in data/processed/04/04_raw_augmented_*.csv; matching row positions and original labels are in 04_row_links_*.csv. Read the CSVs with float_precision='round_trip' and float64 for the three new columns to preserve their exact saved values.

We also appended the new features to the saved Notebook 03 inputs. The old columns remain exactly the same. For its scaled version, total bytes uses log1p, and a separate RobustScaler is fitted only on train. The saved original inputs and raw 45-column inputs remain comparison options.

Saved values, row order, labels, finite values and file hashes passed the checks. All earlier notebooks, raw CSVs, frozen assignments and earlier outputs remain unchanged. Exact matches between the full augmented row sets: 0 summed across representation/row-set pairs. Matches using only the three new columns are reported separately; those columns alone discard most original information. No split was changed. The official test CSV was not loaded; it was only hashed to check preservation.

## Limits and next step

These are summaries of existing measurements, not new information. They can be calculated after a flow is complete if its counters were recorded correctly. They do not prove early detection, botnet-specific labels or independent network context. Source/destination do not necessarily mean client/server. Feature meanings follow the dataset creators' paper, [Tables III and IV](https://doi.org/10.1109/MilCIS.2015.7348942), inspected through [the original paper mirror](https://studylib.net/doc/26070414/07348942). Exact extractor details remain unverified.

Notebook 05 should compare which original and new features are worth using, followed by baseline models. Keep the original 42-column and Notebook 03 inputs as comparisons. Use the fitting rows for learned changes, model_selection for later selection, calibration for later confidence adjustment and the official test only for final evaluation. Check overlap again if columns are removed. No models, feature selection, resampling or confidence calibration were performed here.

## File hashes

- Training CSV: bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa
- Frozen assignments: 2c01281b021a7be536bfd0268d53707a3ea74858730fb577db095d618436b168
- Notebook 03 output list: b88ee5842195ef455d7a5acf9ea92f5768a5b29e3e93b5e4ec9a6097830c4fac

All output hashes are saved in data/processed/04/04_artifact_manifest.json. Detailed checks are kept in results/tables/04_* so the notebook can focus on creating the features.
