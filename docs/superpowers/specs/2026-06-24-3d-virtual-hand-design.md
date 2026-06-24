# 3D virtual hand viewer

**Date:** 2026-06-24
**Status:** Approved, in implementation
**Scope:** Render the decoded gesture as an animated 3D hand in the browser,
fed live by the Rust loop. Phase-1 effector polish; software-only.

## Architecture

- **Rust:** a `WebHand` effector (behind the existing `Effector` trait) owns a
  `PoseBroadcaster` — a small WebSocket server on a background thread using
  `tungstenite` (sync, no async runtime). On each pose it broadcasts a JSON
  line to connected browsers. **Non-blocking:** no client / failed send never
  stalls the deterministic loop; dead clients are dropped.
- **CLI:** `--hand` (optional) selects `WebHand`; `--hand-port` (default
  `8765`). Without `--hand`, the current logging `VirtualHand` is used and the
  loop runs headless exactly as today.
- **Viewer:** static `viewer/hand.html` + `viewer/hand.js` with **vendored**
  three.js (offline-capable). Opened directly in a browser; connects to
  `ws://127.0.0.1:<port>`, renders a procedural hand, and each frame eases the
  finger curl toward the latest closure target.

## The seam (message format)

One JSON message per pose change:

```json
{ "pose": "open", "closure": 0.0 }
```

`closure` ∈ [0,1]: `0` = fully open, `1` = fully closed. Pose→closure table:

| pose   | closure |
|--------|---------|
| open   | 0.0     |
| rest   | 0.15    |
| close  | 1.0     |
| (other)| 0.15    |

The browser interpolates current→target, so the same channel later carries a
continuous `closure` for proportional control — no protocol change.

## Components (isolated, testable)

- `HandPose::closure()` (Rust, in `effector.rs`) — pure pose→closure mapping.
- `PoseBroadcaster` (Rust, new `hand_web.rs`) — owns the accept thread and the
  client list; `broadcast(&str)`. Binds on construction; exposes the bound port
  (allows `:0` for tests).
- `WebHand` (Rust, `hand_web.rs`) — implements `Effector`; on `apply` builds the
  JSON `{pose, closure}` and calls `broadcast`. Also logs, like `VirtualHand`.
- `viewer/hand.{html,js}` — pure front-end: procedural hand geometry, ws client,
  per-frame closure easing.

## Error handling

- `--hand` with a port that fails to bind → clear startup error, the run aborts
  (the user explicitly asked for the hand).
- Per-send failures during the loop are non-fatal: the client is dropped, the
  loop continues. The loop never blocks on the browser.

## Dependencies

- `tungstenite` (sync WebSocket) — browser transport without an async runtime.
- `serde_json` — already present (build the message).

## Testing

- `HandPose::closure()` — table mapping, incl. the `rest`/default case.
- `PoseBroadcaster` — integration test: connect a real `tungstenite` client to
  the bound port, broadcast a message, assert the client receives it (bounded
  retry + read timeout to avoid thread races).
- `WebHand` — `apply` broadcasts a well-formed `{pose, closure}` (verified via a
  connected test client).
- All 27 existing Rust tests stay green; `clippy -D warnings` clean.
- Front-end: manual — run `cargo run -p myo-rt -- --board synthetic --fast
  --model models/lda.json --hand`, open `viewer/hand.html`, watch the hand move.

## Build order (loop runnable throughout)

1. `HandPose::closure()` (TDD).
2. `PoseBroadcaster` + `WebHand` (TDD, real ws client in-test).
3. Wire `--hand` / `--hand-port` into `main.rs`.
4. `viewer/hand.html` + `hand.js` (procedural hand, ws client, easing) — manual
   verify, then update README.
