# Decoder + virtual hand (and the LibEMG trainer that feeds it)

**Date:** 2026-06-23
**Status:** Approved, in implementation
**Scope:** Close the Week-1 loop — classify feature windows with a trained
LDA and drive a virtual hand. Plus the Python `myotrain` package that trains
the model. Software-only, synthetic signal.

## Key decision: native LDA, not the ONNX runtime

LDA is linear: `predict = argmax_k (Wₖ · z + bₖ)`. Python exports the trained
model as a small **model-card JSON**; Rust `decode.rs` loads it and does the
matmul. This is dependency-light, explicitly sanctioned by §4 ("reimplementing
inference directly in Rust is acceptable"), and avoids the `ort` ONNX-runtime
**native library** — the same install risk we already avoid with BrainFlow.
`export_onnx.py` is left as a deferred Phase-4 stub.

## The feature contract (the seam between Python and Rust)

Both sides must produce an identical feature vector:

- Per channel, in this order: `[RMS, MAV, WL, ZC, SSC]`.
- Flattened **channel-major**: `[rms0,mav0,wl0,zc0,ssc0, rms1,…]`, length `5·C`.
- `ZC`/`SSC` use the same noise-deadzone threshold on both sides (in the card).
- Then **z-score standardized** with per-feature mean/std from the training set
  (also in the card). Rust applies `(x − mean)/std` before the linear model.

`FeatureSet` gains `to_vec()` emitting exactly this layout.

## Model-card JSON schema

```json
{
  "model_type": "lda",
  "feature_spec": {
    "features": ["rms","mav","wl","zc","ssc"],
    "channels": 8,
    "order": "channel_major",
    "zc_ssc_threshold": 1e-5
  },
  "standardization": { "mean": [/* 5C */], "std": [/* 5C */] },
  "classes": ["rest","hand_open","hand_close", "…"],
  "weights":   [[/* 5C */], /* one row per class */],
  "intercepts": [/* one per class */]
}
```

`weights` is always **one row per class** (K rows). For a binary model the
exporter expands sklearn's single-row `coef_` into a 2-row argmax form
(`row0 = 0`, `row1 = coef_`), so Rust is always a uniform argmax — no binary
special case.

## Rust components

### `decode.rs`
- `ModelCard` (serde) — deserializes the JSON above.
- `Decoder::load(path)` / `Decoder::from_card(card)`.
- `Decoder::predict(&[f32]) -> Prediction { class_index, label, scores }`:
  validate length == `5·C`; standardize; `scores = W·z + b`; argmax.
- Errors via `MyoError::Decode(String)` (new variant): length mismatch,
  malformed card.

### `effector.rs`
- `Effector` trait: `apply(&mut self, pose: &HandPose)`.
- `HandPose` — minimal: a named pose derived from the predicted class
  (e.g. `rest`, `open`, `close`). Mapping is class-label → pose.
- `VirtualHand` impl: holds current pose, logs transitions via `tracing`.
  (A graphical Python sim is deferred; this closes the loop in-process.)

### `main.rs` wiring
- New optional `--model <path>`.
- No model → current record-only behavior (synthetic stays runnable).
- With model → per window: `FeatureSet::extract` → `to_vec` → `Decoder::predict`
  → map class → `HandPose` → `VirtualHand::apply`. Predicted label logged.

## Python `myotrain` (built after the Rust side is green)

- `python/myotrain/pyproject.toml` — deps: `libemg`, `scikit-learn`, `numpy`,
  `scipy`. Env via `uv`, **Python pinned to 3.12** (3.14 wheels for the
  scientific stack are not reliably available yet).
- `train.py`:
  - Load a **LibEMG built-in dataset** (downloadable; a Myo 8-channel gesture
    set so channel count matches our synthetic default).
  - Window + compute the 5 features in the contract order (LibEMG's
    `FeatureExtractor` provides MAV/RMS/WL/ZC/SSC).
  - `StandardScaler` (capture mean/std) → `LinearDiscriminantAnalysis`.
  - Write the model card JSON.
- `export_onnx.py` — deferred stub (documents the Phase-4 path).

Classification on synthetic *noise* won't be meaningful — that's expected.
This slice proves the **plumbing**: a real trained model loads in Rust, runs on
live windows, and drives the hand. Meaningful accuracy waits for real signal.

## Build order (each step kept green, loop always runnable)

1. `FeatureSet::to_vec` + `decode.rs` (TDD against a hand-written card fixture).
2. `effector.rs` (TDD).
3. Wire into `main.rs` behind `--model`; verify synthetic run with and without
   a fixture card.
4. `myotrain`: env, `train.py`, emit a real card; run the Rust loop against it.

## Testing

- `to_vec`: known window → known ordered vector.
- `decode`: hand-built card → known argmax; length-mismatch errors; binary
  card expands correctly.
- `effector`: class → pose mapping; pose transitions logged.
- `main`: `--model` fixture path drives predictions (smoke run writes parquet
  and logs predicted labels).
- Python: `train.py` on a tiny in-memory set emits a card that `decode.rs`
  round-trips (cross-language contract test, once the env exists).
- `cargo fmt` + `clippy -D warnings` clean throughout.
