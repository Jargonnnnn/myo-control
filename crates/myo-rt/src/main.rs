mod acquire;
mod features;
mod sink;

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use clap::{Parser, ValueEnum};
use thiserror::Error;
use tracing::info;

use acquire::{EmgSource, SyntheticSource};
use features::{FeatureSet, Windower};
use sink::{ParquetSink, SessionMeta};

/// Errors surfaced by the real-time loop. One typed enum at the crate root for
/// now; split per-layer if it grows (see PROJECT.md coding conventions).
#[derive(Debug, Error)]
pub enum MyoError {
    #[error("acquisition error: {0}")]
    Acquire(String),

    #[error("sink error: {0}")]
    Sink(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Board {
    /// Pure-Rust deterministic synthetic source (no hardware, no native deps).
    Synthetic,
}

/// myo-rt: acquisition -> windowing + features -> parquet recorder.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Signal source. Only `synthetic` is wired up in this slice.
    #[arg(long, value_enum, default_value_t = Board::Synthetic)]
    board: Board,

    /// Directory to write the session parquet + sidecar into.
    #[arg(long, default_value = "data/sessions")]
    out: String,

    /// How long to record, in seconds.
    #[arg(long, default_value_t = 2.0)]
    duration: f64,

    /// Number of channels.
    #[arg(long, default_value_t = 8)]
    channels: usize,

    /// Sample rate (Hz).
    #[arg(long, default_value_t = 250)]
    rate: u32,

    /// Window length (ms).
    #[arg(long, default_value_t = 200)]
    window_ms: u32,

    /// Window increment (ms).
    #[arg(long, default_value_t = 50)]
    increment_ms: u32,

    /// Seed for the synthetic source (determinism).
    #[arg(long, default_value_t = 1)]
    seed: u32,

    /// Skip real-time pacing and run as fast as possible.
    #[arg(long)]
    fast: bool,
}

fn ms_to_samples(ms: u32, rate: u32) -> usize {
    (ms as u64 * rate as u64 / 1000) as usize
}

fn main() {
    tracing_subscriber::fmt::init();
    if let Err(e) = run(Args::parse()) {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), MyoError> {
    let window_samples = ms_to_samples(args.window_ms, args.rate).max(1);
    let increment_samples = ms_to_samples(args.increment_ms, args.rate).max(1);
    // One chunk per increment so polls line up with window steps.
    let chunk_samples = increment_samples;

    let mut source = match args.board {
        Board::Synthetic => {
            SyntheticSource::new(args.rate, args.channels, chunk_samples, args.seed)
        }
    };
    // Trust the source for rate/channels; they may differ from CLI defaults
    // once real boards arrive behind the same trait.
    let rate = source.sample_rate_hz();
    let channels = source.channel_count();
    let mut windower = Windower::new(window_samples, increment_samples, channels);

    std::fs::create_dir_all(&args.out)?;
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let session_id = format!("synthetic_{epoch}");
    let meta = SessionMeta {
        session_id: session_id.clone(),
        date: "synthetic".into(),
        subject: "self".into(),
        board_id: "synthetic".into(),
        sample_rate_hz: rate,
        channel_count: channels,
        electrode_placement: "n/a (synthetic)".into(),
        arm_position: "n/a (synthetic)".into(),
        fatigue_state: "n/a (synthetic)".into(),
        gesture_protocol: "default_v1".into(),
        notes: "synthetic source smoke recording".into(),
    };
    let mut sink = ParquetSink::create(std::path::Path::new(&args.out), &meta)?;

    let total_samples = (args.duration * rate as f64) as i64;
    let chunk_dt = Duration::from_secs_f64(chunk_samples as f64 / rate as f64);
    let threshold = 1e-5;

    info!(
        board = ?args.board,
        rate,
        channels,
        window_samples,
        increment_samples,
        total_samples,
        "recording started"
    );

    let mut index: i64 = 0;
    let mut windows_seen: u64 = 0;
    while index < total_samples {
        let chunk = source.poll()?;
        let n = chunk.nrows();
        let t_ms: Vec<i64> = (0..n)
            .map(|i| (index + i as i64) * 1000 / rate as i64)
            .collect();
        sink.write_chunk(&t_ms, chunk.view(), "rest")?;

        for window in windower.push(chunk.view()) {
            let fs = FeatureSet::extract(&window, threshold);
            windows_seen += 1;
            info!(window = windows_seen, rms = ?fs.rms, "features");
        }

        index += n as i64;
        if !args.fast {
            std::thread::sleep(chunk_dt);
        }
    }

    let _ = source.stop();
    let paths = sink.finish()?;
    info!(
        parquet = %paths.parquet.display(),
        meta = %paths.meta.display(),
        windows = windows_seen,
        "recording finished"
    );
    Ok(())
}
