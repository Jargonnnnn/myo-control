//! Decoder: load a trained model card and classify feature vectors.
//!
//! Native LDA — `predict = argmax_k (Wₖ · z + bₖ)` where `z` is the
//! standardized feature vector. No ONNX runtime (see decoder spec). The card
//! is produced by the Python trainer; the feature layout is the shared
//! contract in `features::FeatureSet::to_vec`.

use crate::MyoError;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct FeatureSpec {
    pub features: Vec<String>,
    pub channels: usize,
    pub order: String,
    pub zc_ssc_threshold: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Standardization {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

/// The on-disk trained model, deserialized from JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelCard {
    pub model_type: String,
    pub feature_spec: FeatureSpec,
    pub standardization: Standardization,
    pub classes: Vec<String>,
    /// One row of weights per class, each `5·channels` long.
    pub weights: Vec<Vec<f32>>,
    pub intercepts: Vec<f32>,
}

/// One classification result.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub class_index: usize,
    pub label: String,
    pub scores: Vec<f32>,
}

/// A loaded, validated decoder.
pub struct Decoder {
    card: ModelCard,
}

impl Decoder {
    /// Load and validate a model card from a JSON file.
    pub fn load(path: &Path) -> Result<Self, MyoError> {
        let file = std::fs::File::open(path)?;
        let card: ModelCard = serde_json::from_reader(file)
            .map_err(|e| MyoError::Decode(format!("parsing {}: {e}", path.display())))?;
        Decoder::from_card(card)
    }

    /// Validate the card (schema + dimensions) and wrap it.
    pub fn from_card(card: ModelCard) -> Result<Self, MyoError> {
        if card.model_type != "lda" {
            return Err(MyoError::Decode(format!(
                "unsupported model_type: {}",
                card.model_type
            )));
        }
        if card.feature_spec.order != "channel_major" {
            return Err(MyoError::Decode(format!(
                "unsupported feature order: {} (expected channel_major)",
                card.feature_spec.order
            )));
        }
        const EXPECTED: [&str; 5] = ["rms", "mav", "wl", "zc", "ssc"];
        if card.feature_spec.features.len() != EXPECTED.len()
            || card
                .feature_spec
                .features
                .iter()
                .zip(EXPECTED)
                .any(|(got, want)| got != want)
        {
            return Err(MyoError::Decode(format!(
                "feature set must be {EXPECTED:?}, got {:?}",
                card.feature_spec.features
            )));
        }

        let nf = card.feature_spec.channels * 5;
        let k = card.classes.len();
        if card.standardization.mean.len() != nf || card.standardization.std.len() != nf {
            return Err(MyoError::Decode(format!(
                "standardization length must be {nf} (5 × {} channels)",
                card.feature_spec.channels
            )));
        }
        if card.weights.len() != k {
            return Err(MyoError::Decode(format!(
                "expected {k} weight rows (one per class), got {}",
                card.weights.len()
            )));
        }
        if card.intercepts.len() != k {
            return Err(MyoError::Decode(format!(
                "expected {k} intercepts, got {}",
                card.intercepts.len()
            )));
        }
        if let Some(bad) = card.weights.iter().position(|w| w.len() != nf) {
            return Err(MyoError::Decode(format!(
                "weight row {bad} must have {nf} columns, got {}",
                card.weights[bad].len()
            )));
        }
        if card.standardization.std.contains(&0.0) {
            return Err(MyoError::Decode("standardization std has a zero".into()));
        }
        Ok(Decoder { card })
    }

    /// Expected feature-vector length (`5 × channels`).
    pub fn n_features(&self) -> usize {
        self.card.feature_spec.channels * 5
    }

    /// Channel count the model was trained for.
    pub fn channels(&self) -> usize {
        self.card.feature_spec.channels
    }

    /// The ZC/SSC threshold to extract features with, so the live loop matches
    /// training exactly.
    pub fn zc_ssc_threshold(&self) -> f32 {
        self.card.feature_spec.zc_ssc_threshold
    }

    /// Standardize `x` and return the argmax class.
    pub fn predict(&self, x: &[f32]) -> Result<Prediction, MyoError> {
        let nf = self.n_features();
        if x.len() != nf {
            return Err(MyoError::Decode(format!(
                "expected {nf} features, got {}",
                x.len()
            )));
        }
        let std = &self.card.standardization;
        let z: Vec<f32> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - std.mean[i]) / std.std[i])
            .collect();

        let scores: Vec<f32> = self
            .card
            .weights
            .iter()
            .zip(&self.card.intercepts)
            .map(|(w, b)| w.iter().zip(&z).map(|(wi, zi)| wi * zi).sum::<f32>() + b)
            .collect();

        let class_index = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .ok_or_else(|| MyoError::Decode("no classes in card".into()))?;

        Ok(Prediction {
            class_index,
            label: self.card.classes[class_index].clone(),
            scores,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1-channel card (5 features), identity standardization, two classes.
    // class "b" scores x[0] - 0.5; class "a" scores 0.
    const CARD_JSON: &str = r#"{
        "model_type": "lda",
        "feature_spec": {
            "features": ["rms","mav","wl","zc","ssc"],
            "channels": 1,
            "order": "channel_major",
            "zc_ssc_threshold": 1e-5
        },
        "standardization": { "mean": [0,0,0,0,0], "std": [1,1,1,1,1] },
        "classes": ["a","b"],
        "weights": [[0,0,0,0,0],[1,0,0,0,0]],
        "intercepts": [0.0, -0.5]
    }"#;

    fn decoder() -> Decoder {
        Decoder::from_card(serde_json::from_str(CARD_JSON).unwrap()).unwrap()
    }

    #[test]
    fn predicts_argmax_class() {
        let d = decoder();
        let p = d.predict(&[1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(p.class_index, 1);
        assert_eq!(p.label, "b");

        let p = d.predict(&[0.2, 0.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(p.class_index, 0);
        assert_eq!(p.label, "a");
    }

    #[test]
    fn applies_standardization() {
        // mean=2, std=2 on feature 0: z = (x-2)/2. class b score = z - 0.5.
        // x0 = 6 -> z = 2 -> score_b = 1.5 > 0 -> class b.
        // x0 = 2 -> z = 0 -> score_b = -0.5 < 0 -> class a.
        let json = CARD_JSON.replace(
            r#""mean": [0,0,0,0,0], "std": [1,1,1,1,1]"#,
            r#""mean": [2,0,0,0,0], "std": [2,1,1,1,1]"#,
        );
        let d = Decoder::from_card(serde_json::from_str(&json).unwrap()).unwrap();
        assert_eq!(d.predict(&[6.0, 0.0, 0.0, 0.0, 0.0]).unwrap().label, "b");
        assert_eq!(d.predict(&[2.0, 0.0, 0.0, 0.0, 0.0]).unwrap().label, "a");
    }

    #[test]
    fn rejects_wrong_feature_length() {
        let d = decoder();
        assert!(d.predict(&[1.0, 0.0]).is_err());
    }

    #[test]
    fn n_features_is_five_times_channels() {
        assert_eq!(decoder().n_features(), 5);
    }

    #[test]
    fn rejects_malformed_card_dims() {
        // 3 classes but only 2 weight rows.
        let json = CARD_JSON.replace(r#""classes": ["a","b"]"#, r#""classes": ["a","b","c"]"#);
        let card = serde_json::from_str(&json).unwrap();
        assert!(Decoder::from_card(card).is_err());
    }

    #[test]
    fn rejects_unknown_model_type() {
        let json = CARD_JSON.replace(r#""model_type": "lda""#, r#""model_type": "svm""#);
        let card = serde_json::from_str(&json).unwrap();
        assert!(Decoder::from_card(card).is_err());
    }

    #[test]
    fn rejects_wrong_feature_order() {
        let json = CARD_JSON.replace(r#""order": "channel_major""#, r#""order": "feature_major""#);
        let card = serde_json::from_str(&json).unwrap();
        assert!(Decoder::from_card(card).is_err());
    }

    #[test]
    fn rejects_wrong_feature_names() {
        let json = CARD_JSON.replace(
            r#""features": ["rms","mav","wl","zc","ssc"]"#,
            r#""features": ["rms","mav","wl","zc","var"]"#,
        );
        let card = serde_json::from_str(&json).unwrap();
        assert!(Decoder::from_card(card).is_err());
    }
}
