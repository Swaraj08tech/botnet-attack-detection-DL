# Feature semantics evidence for Notebook 02

Sources checked on 2026-10-11. This review addresses dataset semantics; it does not reproduce the project's separate reference architecture.

## Primary sources

1. [UNSW official dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset): documents the supplied split sizes and nine attack families. Matching counts and names do not authenticate our local bytes against an official checksum.
2. Moustafa and Slay (2015), *UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems*, [DOI](https://doi.org/10.1109/MilCIS.2015.7348942). Tables III, V, VI and VII were inspected in a [readable copy of the original paper](https://studylib.net/doc/26070414/07348942). This is the dataset creators' paper, accessed through a third-party mirror; the publisher endpoint was unavailable.
3. [Scikit-learn StratifiedGroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html): disjoint groups with approximate class stratification. The installed version is recorded with the executed notebook.

## Documented meanings

Table VII defines label 0 as normal and 1 as attack. Table III lists service names and `-` for other cases; this does not establish why a particular flow received that marker. Table VI describes `is_ftp_login` as a binary credential-login indicator and `ct_ftp_cmd` as an FTP-command flow count. The same table describes seven connection counters (`ct_srv_src`, `ct_srv_dst`, `ct_dst_ltm`, `ct_src_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`) over 100 connections ordered using last-time information. It separately describes state/TTL counts and HTTP-method flow counts. Table III gives `sload`/`dload` in bits per second, and Table V describes handshake/interarrival timings.

## Inferences and unresolved questions

- The seven connection counters require state shared across flows. To claim causal deployment, the extractor must specify a past-only window, current-flow inclusion, tie handling and state resets. Those implementation details have not been verified for the supplied CSVs.
- The CSVs lack the endpoints and explicit timestamps needed to replay these counters or isolate hosts, sessions, capture periods and overlapping contextual windows. Grouping exact vectors cannot resolve those dependencies.
- Completed-flow totals are usable in a post-flow benchmark. Availability of the same values for early detection must be independently demonstrated. A prediction-time contract remains required.
- Values 2 and 4 in the nominally binary FTP field and equality with the command counter require extraction/version provenance review. Neither corruption nor label leakage is proven by that observation.
- Strong association with a target is not evidence that a feature was computed from labels. These data support association and consistency checks, not a complete lineage audit.
- The official SharePoint download link redirected to authentication during this review. No official feature-definition CSV or extractor implementation was downloaded. The creator's thesis endpoint also failed to load. The cited paper supports intended definitions, not byte-level correctness of this local subset.
- Original-paper names sometimes differ from subset names. Similar names alone are not a verified mapping for every one of the 42 predictors. No DNS-specific modality, chronological sequence or botnet ground truth is established here.
