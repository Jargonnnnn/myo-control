# myo-control

A non-invasive **myoelectric control testbed**: read surface EMG from the
forearm, decode movement intent, drive an effector (virtual hand first,
physical hand later), and close a tactile feedback loop back to the skin.

This is Phase 0 of a long-horizon neural-integration limb project. The
research spine is **decoder drift robustness** — quantifying and mitigating how
an EMG decoder degrades across days as electrode placement, arm position, and
fatigue vary, and publishing the resulting multi-day dataset.

See [`PROJECT.md`](PROJECT.md) for the full design, architecture, and
constraints.

> ## ⚠️ NOT A MEDICAL DEVICE
>
> Experimental research conducted by the maintainer on themselves. **Not for
> clinical, diagnostic, or third-party use.** No warranty. Anyone building
> body-worn or stimulation hardware from this assumes all regulatory and
> safety responsibility. Anything electrically connected to a human body must
> be battery-powered and isolated.

## Status

Phase 0, first slice: the real-time crate `myo-rt` runs end-to-end on a
**synthetic** signal source (no hardware required):

```
synthetic EMG → windowing → time-domain features (RMS, MAV, WL, ZC, SSC)
              → parquet recording (+ .meta.json sidecar)
```

No decoder, real board, or effector yet — see the roadmap in `PROJECT.md`.

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

## Repository layout

```
crates/myo-rt/     Rust real-time control loop (acquisition → features → effector)
data/              recording schema + protocol (raw recordings gitignored)
docs/              architecture notes, design specs
PROJECT.md         full project context and constraints
```

Future trees (`python/myotrain`, `firmware/`, `hardware/`) arrive with later
phases.

## Licensing

- Code: **Apache-2.0** (see [`LICENSE`](LICENSE)).
- Dataset: **CC-BY** (published separately).
- Hardware/CAD: **CERN-OHL** or **CC-BY**.

Built on the field's open infrastructure (BrainFlow, LibEMG, OpenBCI,
HACKberry/InMoov) — cite it.
