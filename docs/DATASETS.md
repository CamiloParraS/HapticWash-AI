# Datasets

Per-dataset provenance, licence, and citation (SPEC §5, §11). **Datasets are never
committed to either repo. Do not redistribute.** `data/` is gitignored;
`scripts/fetch_data.py` downloads and checksums each set.

Licence verified = a maintainer has read the source's licence page and confirmed it
permits this project's use (non-commercial research, on-device model trained from it,
no redistribution of raw data). Until every row below is verified, downstream work
that depends on the set is on hold (SPEC §11).

| key | licence | verified | role |
|---|---|---|---|
| `zhang_who` | **CC-BY-NC-ND-4.0** | ✅ 2026-09-09 (RDR citation page) | Primary — only public WHO step labels |
| `ablutomania` | CC-BY-4.0 | ✅ 2026-09-22 (Zenodo record) | Hard negatives / confounders |
| `ocdetect` | CC-BY-4.0 | ✅ 2026-09-22 (Zenodo record) | All-day background & NULL for spotting (M5) |
| `harage` | none — not publicly released | ❌ | Supplementary handwashing — proposed drop |

## ⚠️ Open licence blocker — `zhang_who`

`zhang_who` is **CC-BY-NC-ND-4.0**: Attribution + **NonCommercial** + **NoDerivatives**.
It is the *primary* dataset (the only public source with WHO step labels), so this
blocks the project's end goal until resolved. Decision needed from the owner — see
`DECISIONS.md` D4. Two clauses to clear:

- **NC** — fine while HapticWash is a non-commercial portfolio project; violated if it
  ships as a paid/monetised app.
- **ND** — ambiguous for ML. Training model weights from the data is arguably not a
  "derivative" in copyright terms, but CC's ND clause has no ML carve-out. The
  processed corpus (`data/processed/`) is a derivative and must **not** be
  redistributed (the SPEC already forbids this).

Do not start M2 modelling that depends on `zhang_who` until this is decided and
recorded.

---

## `zhang_who`

- **Title:** "Replication Data for: handwashing steps with IMU signals"
- **Author:** Zhang, Yiyuan (contact: Bart Vanrumste), KU Leuven
- **Year / version:** 2022, v1.0
- **Repository:** KU Leuven RDR — <https://rdr.kuleuven.be/citation?persistentId=doi:10.48804/XHPPC7>
- **DOI:** `10.48804/XHPPC7`
- **Licence:** CC-BY-NC-ND-4.0
- **Citation:** Zhang, Y. (2022). *Replication Data for: handwashing steps with IMU
  signals* (V1.0) [Data set]. KU Leuven RDR. https://doi.org/10.48804/XHPPC7
- **Companion paper:** Zhang et al., *IET Healthcare Technology Letters*,
  doi `10.1049/htl2.12018` — detailed data description.

**Structure as downloaded** (`data/zhang-who/`):

- **PART 1 — "11 Hand Washing Steps"**: 10 participants × 3 repetitions = 30 WHO-guided
  washes. 2 Byteflies sensors (ACC + GYR), **100 Hz**, both wrists. Signals are raw
  integer counts, one file per axis (`RAW/ACM_{X,Y,Z}_{L,R}<n>.csv`,
  `GRAW/GYR_..._<n>.csv`), 2-line header (`#Epoch Timestamp`, `#Sampling Rate`).
  Scaling: `ACC_g = raw / 4096`, `GYR_dps = raw / 16.384`. Files `1–3` = participant 1,
  `4–6` = participant 2, … `28–30` = participant 10.
  Per-step annotation in `FRAME/FrameACM_<k>.xls` (k = 1..30), columns
  `Action | Start time | # | End time | #`, where `#` is the sample index (centiseconds
  from `#Epoch Timestamp`). `Action` codes: `0, 0.5, 1, 2.1, 2.2, 3, 4, 5.1, 5.2, 6.1,
  6.2, 7` — finer than canonical §1.3, collapsed by `docs/label_mapping.md` (M1).
- **PART 2 — "8 ADLs"**: 8 participants. ADLs (sitting, typing, standing, walking,
  stairs, teeth) + *untrained* handwashing ×3 (no step breakdown). One CSV per
  wrist/sensor, columns `time,channel1,channel2,channel3` (`time` in seconds).
  Annotation in `8 ADLs shared in RDR/annotation.xlsx`, one sheet per participant, with
  wall-clock + epoch start/end per activity. Feeds NULL / confounders, not step training.

## `ablutomania`

- **Source:** Ablutomania-Set — Scholl, P. et al. Zenodo, DOI `10.5281/zenodo.20094015`
  (<https://zenodo.org/records/20094015>); project page
  <https://earth.informatik.uni-freiburg.de/ablutomania/>.
- **Licence:** CC-BY-4.0 (Zenodo record, checked 2026-09-22).
- **Content:** 22 participants × 2 sessions, 249 wash procedures (~6.5 h): one natural
  wash + five scripted compulsive variants each. Wrist IMU + video of the sink.
- **Files:** 40.6 GB total. Only `handwashing-2019.zip` (2.5 GB), `handwashing-2020.zip`
  (1.6 GB), `hwseminar-2019.zip` (286 MB) and the readme are needed for M1;
  `longterm-2020.zip` (35.7 GB) is all-day data — skip until M5.
- **Citation:** Scholl, P. M., Wahl, K. et al. *Ablutomania-Set* [Data set]. Zenodo.
  https://doi.org/10.5281/zenodo.20094015

## `ocdetect`

- **Source:** OCDetect — Zenodo record `13924901`; pipeline
  <https://github.com/OCDetect/OCDetect-pipeline>. 22 participants, ~2600 h all-day
  Wear OS, 50 Hz acc+gyro, ~2930 labelled washes.
- **Licence:** CC-BY-4.0 (Zenodo record, checked 2026-09-22).
- **Files:** single `OCDetect_dataset.zip`, 31.6 GB, MD5 `897d665e6f9c6f5fbd302e08362baad0`.
  Only needed for M5 — don't download for M1.
- **Labels:** each wash labelled routine vs. compulsive; no WHO step labels.
- **Citation:** Burchard, R., Kirsten, K., Miché, M., Scholl, P., Arnrich, B.,
  Van Laerhoven, K., Lieb, R., Wahl, K. *A Real-World Dataset for detecting Handwashing in
  daily Life using Wrist Motion Data from Wearables.* Scientific Data (2026).
  https://www.nature.com/articles/s41597-026-06698-2 · data DOI `10.5281/zenodo.13924901`

## `harage`

- **Source:** harAGE — Mallol-Ragolta, A., Semertzidou, A., Pateraki, M., Schuller, B.
  *harAGE: A Novel Multimodal Smartwatch-based Dataset for Human Activity Recognition*,
  IEEE FG 2021. ~53 min handwashing, 18 participants.
- **Licence:** none — **no public download.** The paper says data is "available by the
  authors" on request.
- **Why drop:** accelerometer only at 25 Hz (no gyro → can't fill `gx,gy,gz` in §8.1),
  no step labels, and hands were washed *without running water*. Adds little. Only worth
  an email to the authors if M2 is short of subjects.

## `own_phone` / `own_watch`

Self-collected (SPEC §5). No external licence. `own_watch` is evaluation-only and
guarded in code (M6). Not part of the M1 corpus.
