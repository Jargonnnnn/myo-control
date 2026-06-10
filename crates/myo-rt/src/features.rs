//! Windowing and time-domain feature extraction (RMS, MAV, WL, ZC, SSC).
//!
//! Feature functions operate on a single channel (`&[f32]`). The standard
//! Hudgins set; `zc`/`ssc` take a noise-deadzone threshold so small
//! fluctuations near zero don't inflate counts. Counts are returned as `f32`
//! to keep feature vectors uniformly typed.

use ndarray::ArrayView2;

/// One extracted window of multi-channel signal, stored channel-major so each
/// channel is a contiguous slice ready for the feature functions.
#[derive(Debug, Clone)]
pub struct Window {
    channels: Vec<Vec<f32>>,
}

impl Window {
    /// Samples for channel `c`.
    pub fn channel(&self, c: usize) -> &[f32] {
        &self.channels[c]
    }

    /// Number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }
}

/// Accumulates incoming multi-channel samples and emits fixed-length
/// overlapping windows. `window_len` and `increment` are in samples.
pub struct Windower {
    window_len: usize,
    increment: usize,
    buffers: Vec<std::collections::VecDeque<f32>>,
}

impl Windower {
    pub fn new(window_len: usize, increment: usize, channels: usize) -> Self {
        assert!(window_len > 0, "window_len must be > 0");
        assert!(increment > 0, "increment must be > 0");
        assert!(channels > 0, "channels must be > 0");
        Windower {
            window_len,
            increment,
            buffers: vec![std::collections::VecDeque::new(); channels],
        }
    }

    /// Append a chunk of shape `[n_samples, n_channels]` and return any
    /// windows that became complete.
    pub fn push(&mut self, samples: ArrayView2<f32>) -> Vec<Window> {
        for row in samples.rows() {
            for (c, &v) in row.iter().enumerate() {
                self.buffers[c].push_back(v);
            }
        }

        let mut out = Vec::new();
        while self.buffers[0].len() >= self.window_len {
            let channels = self
                .buffers
                .iter()
                .map(|b| b.iter().take(self.window_len).copied().collect())
                .collect();
            out.push(Window { channels });
            for b in &mut self.buffers {
                b.drain(..self.increment);
            }
        }
        out
    }
}

/// The standard Hudgins time-domain feature set, one value per channel.
#[derive(Debug, Clone)]
pub struct FeatureSet {
    pub rms: Vec<f32>,
    pub mav: Vec<f32>,
    pub wl: Vec<f32>,
    pub zc: Vec<f32>,
    pub ssc: Vec<f32>,
}

impl FeatureSet {
    /// Compute every feature for every channel of `window`. `threshold` is the
    /// noise deadzone passed to `zc`/`ssc`.
    pub fn extract(window: &Window, threshold: f32) -> FeatureSet {
        let n = window.channel_count();
        let mut fs = FeatureSet {
            rms: Vec::with_capacity(n),
            mav: Vec::with_capacity(n),
            wl: Vec::with_capacity(n),
            zc: Vec::with_capacity(n),
            ssc: Vec::with_capacity(n),
        };
        for c in 0..n {
            let ch = window.channel(c);
            fs.rms.push(rms(ch));
            fs.mav.push(mav(ch));
            fs.wl.push(wl(ch));
            fs.zc.push(zc(ch, threshold));
            fs.ssc.push(ssc(ch, threshold));
        }
        fs
    }
}

/// Root mean square.
pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    (sum_sq / x.len() as f32).sqrt()
}

/// Mean absolute value.
pub fn mav(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let sum_abs: f32 = x.iter().map(|&v| v.abs()).sum();
    sum_abs / x.len() as f32
}

/// Waveform length: summed absolute difference between consecutive samples.
pub fn wl(x: &[f32]) -> f32 {
    x.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
}

/// Zero crossings: consecutive samples of opposite sign whose difference
/// magnitude clears `threshold`.
pub fn zc(x: &[f32], threshold: f32) -> f32 {
    x.windows(2)
        .filter(|w| w[0] * w[1] < 0.0 && (w[1] - w[0]).abs() >= threshold)
        .count() as f32
}

/// Slope sign changes: interior points where the slope reverses by more than
/// `threshold`.
pub fn ssc(x: &[f32], threshold: f32) -> f32 {
    x.windows(3)
        .filter(|w| (w[1] - w[0]) * (w[1] - w[2]) >= threshold)
        .count() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Three reference single-channel windows with hand-computed features.
    const CONST: [f32; 4] = [2.0, 2.0, 2.0, 2.0];
    const SQUARE: [f32; 4] = [1.0, -1.0, 1.0, -1.0];
    const RAMP: [f32; 4] = [0.0, 1.0, 2.0, 3.0];

    #[test]
    fn rms_matches_hand_computed() {
        assert_relative_eq!(rms(&CONST), 2.0);
        assert_relative_eq!(rms(&SQUARE), 1.0);
        assert_relative_eq!(rms(&RAMP), 3.5_f32.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn mav_is_mean_absolute_value() {
        assert_relative_eq!(mav(&CONST), 2.0);
        assert_relative_eq!(mav(&SQUARE), 1.0);
        assert_relative_eq!(mav(&RAMP), 1.5);
    }

    #[test]
    fn wl_is_summed_absolute_difference() {
        assert_relative_eq!(wl(&CONST), 0.0);
        assert_relative_eq!(wl(&SQUARE), 6.0);
        assert_relative_eq!(wl(&RAMP), 3.0);
    }

    #[test]
    fn zc_counts_threshold_crossings() {
        let thr = 1e-5;
        assert_relative_eq!(zc(&CONST, thr), 0.0);
        assert_relative_eq!(zc(&SQUARE, thr), 3.0);
        assert_relative_eq!(zc(&RAMP, thr), 0.0);
    }

    #[test]
    fn ssc_counts_slope_sign_changes() {
        let thr = 1e-5;
        assert_relative_eq!(ssc(&CONST, thr), 0.0);
        assert_relative_eq!(ssc(&SQUARE, thr), 2.0);
        assert_relative_eq!(ssc(&RAMP, thr), 0.0);
    }

    use ndarray::array;

    #[test]
    fn windower_emits_overlapping_windows() {
        // One channel, samples 0..8, window=4 increment=2 -> 3 windows.
        let mut w = Windower::new(4, 2, 1);
        let samples = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]];
        let windows = w.push(samples.view());
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].channel(0), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(windows[1].channel(0), &[2.0, 3.0, 4.0, 5.0]);
        assert_eq!(windows[2].channel(0), &[4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn windower_handles_split_pushes() {
        // Same as above but delivered in two chunks -> identical windows.
        let mut w = Windower::new(4, 2, 1);
        let a = array![[0.0], [1.0], [2.0]];
        let b = array![[3.0], [4.0], [5.0], [6.0], [7.0]];
        let mut windows = w.push(a.view());
        assert_eq!(windows.len(), 0); // only 3 samples buffered, none ready
        windows.extend(w.push(b.view()));
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].channel(0), &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn windower_keeps_channels_separate() {
        let mut w = Windower::new(2, 2, 2);
        let samples = array![[10.0, 20.0], [11.0, 21.0]];
        let windows = w.push(samples.view());
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].channel(0), &[10.0, 11.0]);
        assert_eq!(windows[0].channel(1), &[20.0, 21.0]);
    }

    #[test]
    fn feature_set_extracts_per_channel() {
        // ch0 = SQUARE, ch1 = RAMP, as a single 4-sample window.
        let mut w = Windower::new(4, 4, 2);
        let samples = array![
            [SQUARE[0], RAMP[0]],
            [SQUARE[1], RAMP[1]],
            [SQUARE[2], RAMP[2]],
            [SQUARE[3], RAMP[3]],
        ];
        let windows = w.push(samples.view());
        let fs = FeatureSet::extract(&windows[0], 1e-5);

        assert_relative_eq!(fs.rms[0], 1.0);
        assert_relative_eq!(fs.rms[1], 3.5_f32.sqrt(), epsilon = 1e-6);
        assert_relative_eq!(fs.wl[0], 6.0);
        assert_relative_eq!(fs.wl[1], 3.0);
        assert_relative_eq!(fs.zc[0], 3.0);
        assert_relative_eq!(fs.ssc[1], 0.0);
    }
}
