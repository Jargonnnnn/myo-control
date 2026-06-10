//! Parquet recorder for the §8 dataset schema plus its `.meta.json` sidecar.
//!
//! One row per sample: `t_ms` (i64), `ch0…chN` (f32, µV), `label` (string).
//! This is the on-disk format the published drift dataset uses, so the Rust
//! loop doubles as a schema-compatible recorder.

use crate::MyoError;
use arrow::array::{ArrayRef, Float32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use ndarray::ArrayView2;
use parquet::arrow::ArrowWriter;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// §8 session metadata, serialized to `<session_id>.meta.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub session_id: String,
    pub date: String,
    pub subject: String,
    pub board_id: String,
    pub sample_rate_hz: u32,
    pub channel_count: usize,
    pub electrode_placement: String,
    pub arm_position: String,
    pub fatigue_state: String,
    pub gesture_protocol: String,
    pub notes: String,
}

/// Where a finished session was written.
#[derive(Debug, Clone)]
pub struct SinkPaths {
    pub parquet: PathBuf,
    pub meta: PathBuf,
}

/// Streams sample rows into a parquet file, writing the sidecar on `finish`.
pub struct ParquetSink {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
    channels: usize,
    paths: SinkPaths,
    meta: SessionMeta,
}

impl ParquetSink {
    /// Open `<dir>/<session_id>.parquet` for writing, using the channel count
    /// from `meta`.
    pub fn create(dir: &Path, meta: &SessionMeta) -> Result<Self, MyoError> {
        let channels = meta.channel_count;
        let mut fields = Vec::with_capacity(channels + 2);
        fields.push(Field::new("t_ms", DataType::Int64, false));
        for c in 0..channels {
            fields.push(Field::new(format!("ch{c}"), DataType::Float32, false));
        }
        fields.push(Field::new("label", DataType::Utf8, false));
        let schema = Arc::new(Schema::new(fields));

        let parquet = dir.join(format!("{}.parquet", meta.session_id));
        let meta_path = dir.join(format!("{}.meta.json", meta.session_id));
        let file = File::create(&parquet)?;
        let writer = ArrowWriter::try_new(file, schema.clone(), None)
            .map_err(|e| MyoError::Sink(e.to_string()))?;

        Ok(ParquetSink {
            writer,
            schema,
            channels,
            paths: SinkPaths {
                parquet,
                meta: meta_path,
            },
            meta: meta.clone(),
        })
    }

    /// Append one chunk: `t_ms` length must equal `samples.nrows()`, and
    /// `samples` must have `channel_count` columns. All rows share `label`.
    pub fn write_chunk(
        &mut self,
        t_ms: &[i64],
        samples: ArrayView2<f32>,
        label: &str,
    ) -> Result<(), MyoError> {
        let n = samples.nrows();
        if t_ms.len() != n {
            return Err(MyoError::Sink(format!(
                "t_ms length {} != sample rows {n}",
                t_ms.len()
            )));
        }
        if samples.ncols() != self.channels {
            return Err(MyoError::Sink(format!(
                "expected {} channels, got {}",
                self.channels,
                samples.ncols()
            )));
        }

        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.channels + 2);
        columns.push(Arc::new(Int64Array::from(t_ms.to_vec())));
        for c in 0..self.channels {
            let col: Vec<f32> = samples.column(c).iter().copied().collect();
            columns.push(Arc::new(Float32Array::from(col)));
        }
        columns.push(Arc::new(StringArray::from(vec![label; n])));

        let batch = RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| MyoError::Sink(e.to_string()))?;
        self.writer
            .write(&batch)
            .map_err(|e| MyoError::Sink(e.to_string()))?;
        Ok(())
    }

    /// Close the parquet writer and write the metadata sidecar.
    pub fn finish(self) -> Result<SinkPaths, MyoError> {
        self.writer
            .close()
            .map_err(|e| MyoError::Sink(e.to_string()))?;
        let file = File::create(&self.paths.meta)?;
        serde_json::to_writer_pretty(file, &self.meta)
            .map_err(|e| MyoError::Sink(e.to_string()))?;
        Ok(self.paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;

    fn sample_meta(channels: usize) -> SessionMeta {
        SessionMeta {
            session_id: "2026-06-10T18-00_s01".into(),
            date: "2026-06-10".into(),
            subject: "self".into(),
            board_id: "synthetic".into(),
            sample_rate_hz: 250,
            channel_count: channels,
            electrode_placement: "n/a (synthetic)".into(),
            arm_position: "n/a (synthetic)".into(),
            fatigue_state: "n/a (synthetic)".into(),
            gesture_protocol: "default_v1".into(),
            notes: "test".into(),
        }
    }

    #[test]
    fn parquet_round_trips_samples() {
        let dir = tempfile::tempdir().unwrap();
        let meta = sample_meta(2);
        let mut sink = ParquetSink::create(dir.path(), &meta).unwrap();
        sink.write_chunk(&[0, 4], array![[1.0, 2.0], [3.0, 4.0]].view(), "rest")
            .unwrap();
        let paths = sink.finish().unwrap();

        let file = File::open(&paths.parquet).unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        // Columns: t_ms, ch0, ch1, label.
        assert_eq!(batch.num_columns(), 4);
        assert_eq!(batch.num_rows(), 2);
        let schema = batch.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["t_ms", "ch0", "ch1", "label"]);

        use arrow::array::{Float32Array, Int64Array, StringArray};
        let t = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(t.values(), &[0, 4]);
        let ch1 = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(ch1.values(), &[2.0, 4.0]);
        let label = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(label.value(0), "rest");
    }

    #[test]
    fn sidecar_round_trips_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let meta = sample_meta(8);
        let sink = ParquetSink::create(dir.path(), &meta).unwrap();
        let paths = sink.finish().unwrap();

        let read: SessionMeta = serde_json::from_reader(File::open(&paths.meta).unwrap()).unwrap();
        assert_eq!(read.session_id, "2026-06-10T18-00_s01");
        assert_eq!(read.channel_count, 8);
        assert_eq!(read.board_id, "synthetic");
    }
}
