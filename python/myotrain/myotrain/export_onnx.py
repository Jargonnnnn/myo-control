"""Deferred: ONNX export path.

The Week-1 decoder uses a native-LDA model card (JSON) consumed directly by the
Rust loop — no ONNX runtime — per PROJECT.md §4 ("reimplementing inference
directly in Rust is acceptable and dependency-light") and the decoder spec.

ONNX export becomes worthwhile once models stop being trivially linear (e.g.
the Phase-4 nonlinear/ensemble decoders). At that point this module would use
``skl2onnx`` to convert the trained estimator and emit a ``.onnx`` alongside the
feature/standardization metadata. Intentionally not implemented yet.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "export_onnx is a deferred Phase-4 stub; the Week-1 path is the "
        "native-LDA model card written by myotrain.train."
    )


if __name__ == "__main__":
    main()
