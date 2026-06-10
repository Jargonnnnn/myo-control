# Data

Raw recordings live here but are **not committed** (see `.gitignore`). The
curated multi-day drift dataset is published separately under CC-BY.

The authoritative schema and recording protocol are defined in `PROJECT.md` §8.
A fuller version of this README — schema table + protocol — is a Week-4
deliverable. For now:

- One **parquet** file per recording session: columns `t_ms` (int64),
  `ch0…chN` (float32, µV), `label` (categorical gesture id; `rest` is a label).
- One **JSON sidecar** per session (`<session>.meta.json`) with acquisition
  metadata (board, sample rate, channel count, electrode placement, arm
  position, fatigue state, gesture protocol, notes).
