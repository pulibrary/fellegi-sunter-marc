#![allow(dead_code)]

//! Fellegi-Sunter record linkage model for MARC records
//!
//! This library provides functionality to train and execute Fellegi-Sunter models
//! for record linkage based on field probabilities. It accepts vectors of field
//! probabilities rather than actual records, enabling supervised training where
//! users provide the probabilities directly.
//!
//! # Training
//!
//! The model supports supervised training, where users supply probability
//! vectors for matched and unmatched record pairs. Missing fields can be represented
//! by using None values (specifically f64::NAN) to indicate missing data in the field probability.
//!
//! # Usage
//!
//! ```rust
//! use fellegi_sunter_marc::{FellegiSunterModel, FieldProbabilities};
//!
//! // Create a new model with 3 fields
//! let mut model = FellegiSunterModel::new(3);
//!
//! // Train the model with some sample data - including missing values
//! let matched_samples = vec![
//!     FieldProbabilities::new(vec![0.9, 0.8, f64::NAN]), // Third field is missing
//!     FieldProbabilities::new(vec![0.8, 0.9, 0.6]),
//! ];
//!
//! let unmatched_samples = vec![
//!     FieldProbabilities::new(vec![0.1, f64::NAN, 0.3]), // Second field is missing
//!     FieldProbabilities::new(vec![0.2, 0.1, 0.4]),
//! ];
//!
//! model.train(&matched_samples, &unmatched_samples);
//!
//! // Run the model on new data
//! let test_pair = FieldProbabilities::new(vec![0.85, 0.75, 0.65]);
//! let score = model.score(&test_pair);
//! println!("Linkage score: {}", score);
//! ```

mod json;
mod marc;

use std::f64;

pub use json::{ClusterData, TRAINING_CLUSTERS};
pub use marc::{BENCHMARK_MARC, similarities_between_records};

/// Represents a set of field probabilities for a record pair
#[derive(Debug, Clone)]
pub struct FieldProbabilities {
    /// Vector of probabilities for each field. Use f64::NAN to represent missing/None values.
    pub probabilities: Vec<f64>,
}

impl FieldProbabilities {
    /// Create new FieldProbabilities with given probabilities
    pub fn new(probabilities: Vec<f64>) -> Self {
        FieldProbabilities { probabilities }
    }

    /// Get the number of fields
    pub fn len(&self) -> usize {
        self.probabilities.len()
    }

    /// Check if there are no fields (empty)
    pub fn is_empty(&self) -> bool {
        self.probabilities.is_empty()
    }

    /// Check if a field value is missing (is NaN)
    pub fn is_field_missing(&self, index: usize) -> bool {
        index < self.probabilities.len() && self.probabilities[index].is_nan()
    }

    /// Get the probability for a specific field, or None if missing
    pub fn get_field_probability(&self, index: usize) -> Option<f64> {
        if index >= self.probabilities.len() || self.probabilities[index].is_nan() {
            None
        } else {
            Some(self.probabilities[index])
        }
    }
}

/// Fellegi-Sunter model for record linkage  
#[derive(Debug, Clone)]
pub struct FellegiSunterModel {
    /// Number of fields in the model
    num_fields: usize,

    /// Probability that a field matches given records are matched (P(field|match))
    /// This represents parameters we'll estimate during training  
    p_field_match: Vec<f64>,

    /// Probability that a field matches given records are unmatched (P(field|non-match))
    /// This represents parameters we'll estimate during training  
    p_field_non_match: Vec<f64>,

    /// Prior probability of match
    prior_match: f64,
}

impl FellegiSunterModel {
    /// Create a new Fellegi-Sunter model with the specified number of fields
    pub fn new(num_fields: usize) -> Self {
        FellegiSunterModel {
            num_fields,
            p_field_match: vec![0.5; num_fields], // Initialize to 0.5
            p_field_non_match: vec![0.5; num_fields], // Initialize to 0.5
            prior_match: 0.5,                     // Initialize to 0.5
        }
    }

    /// Train the model using supervised learning with provided matched and unmatched samples
    ///
    /// # Arguments
    /// * `matched_samples` - Vector of field probability vectors for pairs that are known to match
    /// * `unmatched_samples` - Vector of field probability vectors for pairs that are known not to match
    pub fn train(
        &mut self,
        matched_samples: &[FieldProbabilities],
        unmatched_samples: &[FieldProbabilities],
    ) {
        // Validate inputs
        if matched_samples.is_empty() && unmatched_samples.is_empty() {
            return; // Nothing to train on
        }

        // Update field probabilities for matched pairs
        for i in 0..self.num_fields {
            let mut matched_probs: Vec<f64> = Vec::new();
            let mut unmatched_probs: Vec<f64> = Vec::new();

            // Collect probabilities from matched samples for this field
            for sample in matched_samples.iter() {
                if i < sample.probabilities.len() && !sample.probabilities[i].is_nan() {
                    matched_probs.push(sample.probabilities[i]);
                }
            }

            // Collect probabilities from unmatched samples for this field
            for sample in unmatched_samples.iter() {
                if i < sample.probabilities.len() && !sample.probabilities[i].is_nan() {
                    unmatched_probs.push(sample.probabilities[i]);
                }
            }

            // Estimate P(field|match) - mean of probabilities for matched pairs (excluding missing values)
            let mean_matched = if !matched_probs.is_empty() {
                matched_probs.iter().sum::<f64>() / matched_probs.len() as f64
            } else {
                0.5
            };
            self.p_field_match[i] = mean_matched;

            // Estimate P(field|non-match) - mean of probabilities for unmatched pairs (excluding missing values)
            let mean_unmatched = if !unmatched_probs.is_empty() {
                unmatched_probs.iter().sum::<f64>() / unmatched_probs.len() as f64
            } else {
                0.5
            };
            self.p_field_non_match[i] = mean_unmatched;
        }

        // Update prior probability of match based on sample sizes
        if !matched_samples.is_empty() && !unmatched_samples.is_empty() {
            self.prior_match = matched_samples.len() as f64
                / (matched_samples.len() + unmatched_samples.len()) as f64;
        } else if !matched_samples.is_empty() {
            self.prior_match = 1.0;
        } else if !unmatched_samples.is_empty() {
            self.prior_match = 0.0;
        }
    }

    /// Calculate the linkage score for a given pair of field probabilities
    ///
    /// # Arguments
    /// * `probabilities` - Field probability vector for the record pair to score
    ///
    /// # Returns
    /// The normalized log-likelihood ratio (score) indicating how likely this is a match
    pub fn score(&self, probabilities: &FieldProbabilities) -> f64 {
        // Check that we have the right number of fields
        if probabilities.probabilities.len() != self.num_fields {
            panic!(
                "Probability vector has {} fields, but model expects {}",
                probabilities.probabilities.len(),
                self.num_fields
            );
        }

        // Calculate likelihood ratio for each field (excluding missing fields)
        let mut log_likelihood_ratio = 0.0;
        let mut field_count = 0; // Count of fields that are not missing

        for i in 0..self.num_fields {
            // Skip missing fields
            if probabilities.probabilities[i].is_nan() {
                continue;
            }

            let p_field_match = self.p_field_match[i];
            let p_field_non_match = self.p_field_non_match[i];

            // Calculate ln(P(field|match)/P(field|non-match))
            // Handle edge cases for numerical stability
            let likelihood_ratio = if p_field_non_match > 0.0 && p_field_match > 0.0 {
                p_field_match / p_field_non_match
            } else if p_field_match > 0.0 {
                f64::INFINITY
            } else if p_field_non_match > 0.0 {
                0.0
            } else {
                1.0 // Both are zero - no discrimination information
            };

            let log_ratio = if likelihood_ratio > 0.0 {
                likelihood_ratio.ln()
            } else {
                f64::NEG_INFINITY
            };

            log_likelihood_ratio += log_ratio;
            field_count += 1;
        }

        // Apply prior probability adjustment
        let prior_ratio = if self.prior_match > 0.0 && (1.0 - self.prior_match) > 0.0 {
            self.prior_match / (1.0 - self.prior_match)
        } else if self.prior_match > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Return final score; normalize by number of fields compared to total available fields
        let normalized_score = log_likelihood_ratio + prior_ratio.ln();

        // Adjust for missing fields - if all fields are missing, return a default value
        if field_count == 0 {
            0.0 // No actual comparisons were made
        } else {
            normalized_score
        }
    }

    /// Get the current probability of matching for each field
    pub fn get_p_field_match(&self) -> &[f64] {
        &self.p_field_match
    }

    /// Get the current probability of non-matching for each field  
    pub fn get_p_field_non_match(&self) -> &[f64] {
        &self.p_field_non_match
    }

    /// Get the current prior probability of match
    pub fn get_prior_match(&self) -> f64 {
        self.prior_match
    }
}
