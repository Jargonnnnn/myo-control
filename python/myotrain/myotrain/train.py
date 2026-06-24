"""Train a baseline LDA and export a model card for the Rust decoder.

Week-1 scope: this trains on a small **synthetic, class-separable** dataset so
the full train -> card -> Rust-decode plumbing is provable end-to-end with no
network or hardware. Classification of *real* intent waits for real signal;
LibEMG datasets + robustness feature sets arrive with the drift-study slice.

The exported card matches ``crates/myo-rt/src/decode.rs::ModelCard``:

    model_type, feature_spec{features, channels, order, zc_ssc_threshold},
    standardization{mean, std}, classes, weights (one row per class), intercepts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from . import features

CLASSES = ["rest", "hand_open", "hand_close"]
ZC_SSC_THRESHOLD = 1e-5


def synthesize(channels: int, window: int, per_class: int, seed: int):
    """Generate separable synthetic windows: each gesture elevates EMG amplitude
    on a distinct channel group, so the time-domain features separate."""
    rng = np.random.default_rng(seed)
    half = channels // 2
    # Per-class amplitude profile across channels (microvolt-ish std).
    profiles = {
        "rest": np.full(channels, 5.0),
        "hand_open": np.where(np.arange(channels) < half, 40.0, 5.0),
        "hand_close": np.where(np.arange(channels) >= half, 40.0, 5.0),
    }
    feats: list[list[float]] = []
    labels: list[int] = []
    for ci, name in enumerate(CLASSES):
        amp = profiles[name]
        for _ in range(per_class):
            win = rng.standard_normal((window, channels)) * amp
            feats.append(features.window_features(win, ZC_SSC_THRESHOLD))
            labels.append(ci)
    return np.asarray(feats, dtype=np.float64), np.asarray(labels)


def build_card(
    scaler: StandardScaler, lda: LinearDiscriminantAnalysis, channels: int
) -> dict:
    coef = np.asarray(lda.coef_, dtype=np.float64)
    intercept = np.asarray(lda.intercept_, dtype=np.float64)
    nf = channels * 5
    # sklearn gives a single row for the binary case; expand to a uniform
    # one-row-per-class argmax form so the Rust side never special-cases it.
    if coef.shape[0] == 1:
        weights = [[0.0] * nf, coef[0].tolist()]
        intercepts = [0.0, float(intercept[0])]
    else:
        weights = coef.tolist()
        intercepts = intercept.tolist()
    return {
        "model_type": "lda",
        "feature_spec": {
            "features": features.FEATURES,
            "channels": channels,
            "order": "channel_major",
            "zc_ssc_threshold": ZC_SSC_THRESHOLD,
        },
        "standardization": {
            "mean": scaler.mean_.tolist(),
            "std": scaler.scale_.tolist(),
        },
        "classes": CLASSES,
        "weights": weights,
        "intercepts": intercepts,
    }


def card_predict(card: dict, x: np.ndarray) -> np.ndarray:
    """Reproduce the Rust decoder's math: standardize, then argmax(W·z + b)."""
    mean = np.asarray(card["standardization"]["mean"])
    std = np.asarray(card["standardization"]["std"])
    w = np.asarray(card["weights"])
    b = np.asarray(card["intercepts"])
    z = (x - mean) / std
    return (z @ w.T + b).argmax(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train baseline LDA, export model card.")
    ap.add_argument("--out", default="models/lda.json", help="output card path")
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--window", type=int, default=50, help="samples per window")
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    x, y = synthesize(args.channels, args.window, args.per_class, args.seed)
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    lda = LinearDiscriminantAnalysis().fit(xs, y)

    train_acc = float((lda.predict(xs) == y).mean())
    card = build_card(scaler, lda, args.channels)

    # Guard: the exported card must reproduce sklearn's own predictions exactly,
    # otherwise the Rust decoder would disagree with training.
    mismatches = int((card_predict(card, x) != lda.predict(xs)).sum())
    if mismatches:
        raise SystemExit(f"card/model disagree on {mismatches} samples — aborting")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2))
    print(
        f"wrote {out}  classes={CLASSES}  channels={args.channels} "
        f"features={len(card['standardization']['mean'])}  train_acc={train_acc:.3f}"
    )


if __name__ == "__main__":
    main()
