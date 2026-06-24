//! Acquisition: the `EmgSource` trait and a pure-Rust `SyntheticSource`.
//!
//! `EmgSource` is the seam every signal origin sits behind — synthetic now,
//! BrainFlow (synthetic board, playback, real boards) later. Keeping the trait
//! here means the loop never names a concrete source.

use crate::MyoError;
use ndarray::Array2;

/// A source of multi-channel EMG samples.
pub trait EmgSource {
    fn sample_rate_hz(&self) -> u32;
    fn channel_count(&self) -> usize;

    /// Pull the next chunk of samples, shape `[n_samples, n_channels]`. A real
    /// board blocks until samples arrive; the synthetic source returns
    /// immediately and is paced by the loop.
    fn poll(&mut self) -> Result<Array2<f32>, MyoError>;

    /// Release any underlying resources. Idempotent.
    fn stop(&mut self) -> Result<(), MyoError>;
}

/// Deterministic synthetic EMG: per-channel band-limited noise from a seeded
/// xorshift, so a given seed always yields the same stream (reproducible
/// tests, no hardware, no native deps). Not a board driver — a test source.
pub struct SyntheticSource {
    sample_rate_hz: u32,
    channels: usize,
    chunk_samples: usize,
    /// xorshift32 state.
    state: u32,
    /// One-pole low-pass state per channel (band-limits the white noise).
    lp: Vec<f32>,
    /// When set, modulate per-channel amplitude to cycle rest/open/close so a
    /// trained decoder produces visible gesture changes (demo aid, not a board).
    gesture_demo: bool,
    /// Samples emitted so far (drives the gesture cycle).
    elapsed: u64,
}

/// Per-channel amplitude for the gesture-demo cycle at `elapsed` samples:
/// rest → open → close, two seconds each. `open` elevates the first half of
/// the channels, `close` the second half — matching the trainer's profiles.
fn gesture_demo_amplitudes(elapsed: u64, sample_rate_hz: u32, channels: usize) -> Vec<f32> {
    let phase = (elapsed / (sample_rate_hz as u64 * 2)) % 3;
    let half = channels / 2;
    (0..channels)
        .map(|c| match phase {
            0 => 5.0,               // rest
            1 if c < half => 40.0,  // open
            2 if c >= half => 40.0, // close
            _ => 5.0,
        })
        .collect()
}

impl SyntheticSource {
    pub fn new(sample_rate_hz: u32, channels: usize, chunk_samples: usize, seed: u32) -> Self {
        assert!(channels > 0, "channels must be > 0");
        assert!(chunk_samples > 0, "chunk_samples must be > 0");
        SyntheticSource {
            sample_rate_hz,
            channels,
            chunk_samples,
            // Avoid a zero state, which xorshift cannot escape.
            state: seed | 1,
            lp: vec![0.0; channels],
            gesture_demo: false,
            elapsed: 0,
        }
    }

    /// Enable the gesture-demo cycle (builder-style).
    pub fn with_gesture_demo(mut self, on: bool) -> Self {
        self.gesture_demo = on;
        self
    }

    /// xorshift32 -> uniform noise in [-1.0, 1.0).
    fn next_noise(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        (x as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

impl EmgSource for SyntheticSource {
    fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    fn channel_count(&self) -> usize {
        self.channels
    }

    fn poll(&mut self) -> Result<Array2<f32>, MyoError> {
        // Band-limited noise, distinct per channel via the running filter state.
        // Amplitude is a flat ~50 µV, or the gesture-cycle profile when demoing.
        const ALPHA: f32 = 0.2; // low-pass smoothing factor
        const SCALE: f32 = 50.0; // microvolts
        let mut out = Array2::<f32>::zeros((self.chunk_samples, self.channels));
        // The one-pole filter attenuates variance (~0.19 std), so demo
        // amplitudes are scaled up to land in the trainer's feature range.
        const DEMO_GAIN: f32 = 5.3;
        for s in 0..self.chunk_samples {
            let amp = if self.gesture_demo {
                gesture_demo_amplitudes(self.elapsed, self.sample_rate_hz, self.channels)
                    .iter()
                    .map(|a| a * DEMO_GAIN)
                    .collect()
            } else {
                vec![SCALE; self.channels]
            };
            for c in 0..self.channels {
                let n = self.next_noise();
                self.lp[c] += ALPHA * (n - self.lp[c]);
                out[[s, c]] = self.lp[c] * amp[c];
            }
            self.elapsed += 1;
        }
        Ok(out)
    }

    fn stop(&mut self) -> Result<(), MyoError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_reports_config_and_shape() {
        let mut s = SyntheticSource::new(250, 8, 64, 1);
        assert_eq!(s.sample_rate_hz(), 250);
        assert_eq!(s.channel_count(), 8);
        let chunk = s.poll().unwrap();
        assert_eq!(chunk.shape(), &[64, 8]);
    }

    #[test]
    fn gesture_demo_cycles_rest_open_close() {
        let rate = 250;
        let ch = 8;
        let secs = |s: u64| s * rate as u64;
        // rest: all channels low.
        assert_eq!(gesture_demo_amplitudes(0, rate, ch), vec![5.0; 8]);
        // open: first half elevated.
        assert_eq!(
            gesture_demo_amplitudes(secs(2), rate, ch),
            vec![40.0, 40.0, 40.0, 40.0, 5.0, 5.0, 5.0, 5.0]
        );
        // close: second half elevated.
        assert_eq!(
            gesture_demo_amplitudes(secs(4), rate, ch),
            vec![5.0, 5.0, 5.0, 5.0, 40.0, 40.0, 40.0, 40.0]
        );
        // wraps back to rest.
        assert_eq!(gesture_demo_amplitudes(secs(6), rate, ch), vec![5.0; 8]);
    }

    #[test]
    fn synthetic_is_deterministic_for_seed() {
        let mut a = SyntheticSource::new(250, 8, 64, 42);
        let mut b = SyntheticSource::new(250, 8, 64, 42);
        assert_eq!(a.poll().unwrap(), b.poll().unwrap());
    }

    #[test]
    fn synthetic_differs_across_seeds() {
        let mut a = SyntheticSource::new(250, 8, 64, 42);
        let mut c = SyntheticSource::new(250, 8, 64, 7);
        assert_ne!(a.poll().unwrap(), c.poll().unwrap());
    }

    #[test]
    fn synthetic_stream_advances_between_polls() {
        let mut s = SyntheticSource::new(250, 2, 16, 5);
        assert_ne!(s.poll().unwrap(), s.poll().unwrap());
    }
}
