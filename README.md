# myo-control

A non-invasive **myoelectric control testbed**: read surface EMG from the
forearm, decode movement intent, drive an effector (virtual hand first,
physical hand later), and close a tactile feedback loop back to the skin.

This is Phase 0 of a long-horizon neural-integration limb project. The
research spine is **decoder drift robustness** — quantifying and mitigating how
an EMG decoder degrades across days as electrode placement, arm position, and
fatigue vary, and publishing the resulting multi-day dataset.

Design notes and architecture decisions live under [`docs/`](docs/).

> ## ⚠️ NOT A MEDICAL DEVICE
>
> Experimental research conducted by the maintainer on themselves. **Not for
> clinical, diagnostic, or third-party use.** No warranty. Anyone building
> body-worn or stimulation hardware from this assumes all regulatory and
> safety responsibility. Anything electrically connected to a human body must
> be battery-powered and isolated.

## Status

Phase 0: the full Week-1 loop runs end-to-end on a **synthetic** signal source
(no hardware required) — acquisition through to a decoded gesture driving a
virtual hand:

```
synthetic EMG → windowing → time-domain features (RMS, MAV, WL, ZC, SSC)
              → parquet recording (+ .meta.json sidecar)
              → LDA decode (trained model card) → virtual hand
```

A baseline LDA is trained in Python (`myotrain`) on synthetic, separable data
and exported as a model card the Rust loop reads directly (native LDA — no ONNX
runtime). Still to come: a real acquisition path (BrainFlow), real recordings,
proportional control, tactile feedback, and the multi-day drift study.

## Quick start

Develop with no hardware against the synthetic board:

```bash
# Record 2 s of synthetic EMG to data/sessions/ and log feature vectors
cargo run -p myo-rt -- --board synthetic

# Options: --duration <s> --channels <n> --rate <hz> --window-ms --increment-ms --fast
cargo run -p myo-rt -- --board synthetic --duration 5 --channels 8 --fast

cargo test                                  # unit tests
cargo fmt && cargo clippy --all-targets -- -D warnings
```

Recorded sessions land in `data/sessions/` and are **git-ignored** — raw
recordings are never committed (the curated dataset is published separately).

### Close the loop with a trained decoder

Train a baseline LDA (synthetic, separable data for now) and export a model
card, then let the Rust loop decode live windows and drive the virtual hand:

```bash
# One-time: pinned Python env for training
uv venv --python 3.12 python/myotrain/.venv
uv pip install --python python/myotrain/.venv numpy scikit-learn

# Train -> model card (JSON consumed directly by the Rust decoder; no ONNX)
PYTHONPATH=python/myotrain python/myotrain/.venv/bin/python \
    -m myotrain.train --out models/lda.json

# Run the loop with the model: each window is classified and drives the hand
cargo run -p myo-rt -- --board synthetic --fast --model models/lda.json
```

Without `--model` the loop just records; with it, predictions drive the
virtual hand. (Classification on synthetic *noise* is not meaningful — this
proves the train → card → decode plumbing; real signal comes later.)

## Repository layout

```
crates/myo-rt/     Rust real-time control loop (acquisition → features → decode → effector)
python/myotrain/   training + offline analysis (LDA, model-card export)
data/              recording schema + protocol (raw recordings gitignored)
docs/              architecture notes, design specs
```

Future trees (`firmware/`, `hardware/`) arrive with later phases.

## Licensing

- Code: **Apache-2.0** (see [`LICENSE`](LICENSE)).
- Dataset: **CC-BY** (published separately).
- Hardware/CAD: **CERN-OHL** or **CC-BY**.

Built on the field's open infrastructure (BrainFlow, LibEMG, OpenBCI,
HACKberry/InMoov) — cite it.
