# myo-rt acquisition → features → parquet spine

**Date:** 2026-06-09
**Status:** Approved, in implementation
**Scope:** First runnable slice of `myo-rt`. Software-only, no hardware, no ML.

## Goal

Stand up the `myo-rt` Rust crate so that:

```
cargo run -p myo-rt -- --board synthetic
```

pulls EMG windows from a synthetic signal source, runs windowing + the basic
time-domain features (RMS, MAV, WL, ZC, SSC), writes raw samples to a parquet
file in the §8 schema with a `.meta.json` sidecar, and logs the computed
feature vectors. This is the spine everything else (decoder, effector,
feedback, real boards, playback) later hangs off.

## Decisions

- **Trait-first acquisition.** An `EmgSource` trait is the seam. A pure-Rust
  `SyntheticSource` is the first implementation, giving a runnable loop with
  zero native dependencies. BrainFlow remains the intended real acquisition
  path (synthetic board, playback board, real boards) behind the same trait,
  but its native-lib integration is **deferred to a later slice** to avoid
  install risk now. The synthetic generator is a test source, not a board
  driver — it does not violate the "don't write board drivers" rule.
- **Sink writes the §8 raw schema**, not feature vectors. The Rust loop is a
  real, dataset-compatible recorder. Features are computed in-loop and logged
  via `tracing` to prove the stage runs; they are not given a second on-disk
  format yet (YAGNI).
- **Working name stays `myo-control`** — no rename now.

## Crate layout

```
myo-control/
├── Cargo.toml              # workspace
├── .gitignore
├── crates/myo-rt/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs         # CLI (clap) + loop orchestration
│       ├── acquire.rs      # EmgSource trait + SyntheticSource
│       ├── features.rs     # Windower + RMS/MAV/WL/ZC/SSC
│       └── sink.rs         # ParquetSink + .meta.json sidecar
├── data/
│   ├── .gitignore          # ignore raw recordings
│   └── README.md           # stub; full schema/protocol is Week-4
```

`sink.rs` is not in the original project file list; it is added because §8's
parquet + sidecar output warrants its own focused module rather than living in
`main.rs`. `decode.rs`, `effector.rs`, `feedback.rs`, `python/`, and
`firmware/` are out of scope for this slice.

## Interfaces

### `acquire.rs`

```rust
pub trait EmgSource {
    fn sample_rate_hz(&self) -> u32;
    fn channel_count(&self) -> usize;
    /// Pull the next chunk of samples, shape [n_new_samples, n_channels].
    /// Blocks until samples are available.
    fn poll(&mut self) -> Result<Array2<f32>, MyoError>;
    fn stop(&mut self) -> Result<(), MyoError>;
}
```

`SyntheticSource`: seeded, deterministic band-limited noise per channel via a
hand-rolled xorshift + light smoothing (avoids pulling `rand`/`rand_distr`).
Same seed → identical stream. Honours configured sample rate and channel count.

### `features.rs`

- `Windower` — accumulates samples, yields fixed-length overlapping windows.
  Defaults: **200 ms window, 50 ms increment**.
- Per-window feature functions, each returning a per-channel vector:
  - `rms`, `mav`, `wl` (waveform length), `zc` (zero crossings), `ssc` (slope
    sign changes). `zc`/`ssc` use a small noise-deadzone threshold (standard
    Hudgins) so noise near zero doesn't inflate counts.
- `FeatureSet` — bundles the per-channel feature vectors for one window.

### `sink.rs`

- `ParquetSink` — writes the §8 raw schema, one row per sample:
  `t_ms` (i64), `ch0…chN` (f32), `label` (string, `"rest"` for now).
- Writes `<session>.meta.json` sidecar with §8 metadata on flush.

### `main.rs`

CLI via `clap`: `--board synthetic` (only option now), `--out <dir>`,
`--duration <s>`, `--channels`, `--rate`, `--window-ms`, `--increment-ms`.
Loop: `poll()` → write raw samples to sink → feed windower → on each emitted
window compute `FeatureSet` and `tracing::info!` it. Runs for `--duration` or
until Ctrl-C, then flushes parquet + writes the sidecar.

Defaults mirror the §8 meta.json example: 250 Hz, 8 channels.

## Dependencies

- `ndarray`, `clap`, `tracing`, `tracing-subscriber`, `serde`, `serde_json`
  — all sanctioned by the project tech stack.
- `arrow`, `parquet` — write the §8 parquet (spec calls for parquet I/O).
- `thiserror` — typed per-layer error enum (`MyoError`); no `unwrap`/`expect`
  in the loop path.
- No `rand` — synthetic noise is a hand-rolled seeded xorshift.

## Testing (definition of done)

- `features.rs`: known signals → known features (constant → WL≈0, ZC=0; square
  wave → known ZC count; ramp → known MAV).
- `Windower`: correct window count + overlap for a known sample count.
- `SyntheticSource`: same seed → identical output (determinism).
- `sink.rs`: parquet round-trip — write, read back, assert columns + values.
- `cargo fmt` clean; `cargo clippy --all-targets -- -D warnings` clean.

Done = `cargo run -p myo-rt -- --board synthetic` produces a parquet +
sidecar and logs feature vectors, with all tests green.
```
