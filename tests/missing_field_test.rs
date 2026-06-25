//! Tests for missing field handling in Fellegi-Sunter model

use fellegi_sunter_marc::{FellegiSunterModel, FieldProbabilities};

#[test]
fn test_missing_field_handling() {
    let mut model = FellegiSunterModel::new(3);
    
    // Train with some samples that include missing fields
    let matched_samples = vec![
        FieldProbabilities::new(vec![0.9, 0.8, f64::NAN]), // Third field is missing
        FieldProbabilities::new(vec![0.8, 0.9, 0.6]),
    ];

    let unmatched_samples = vec![
        FieldProbabilities::new(vec![0.1, f64::NAN, 0.3]), // Second field is missing
        FieldProbabilities::new(vec![0.2, 0.1, 0.4]),
    ];

    model.train(&matched_samples, &unmatched_samples);
    
    // Test scoring with missing fields
    let test_pair = FieldProbabilities::new(vec![0.85, f64::NAN, 0.65]);
    let score = model.score(&test_pair);
    
    // Score should be a finite number
    assert!(score.is_finite());
    assert!(!score.is_nan());
}

#[test]
fn test_all_fields_missing() {
    let mut model = FellegiSunterModel::new(3);
    
    // Train with samples
    let matched_samples = vec![
        FieldProbabilities::new(vec![0.9, 0.8, 0.7]),
        FieldProbabilities::new(vec![0.8, 0.9, 0.6]),
    ];

    let unmatched_samples = vec![
        FieldProbabilities::new(vec![0.1, 0.2, 0.3]),
        FieldProbabilities::new(vec![0.2, 0.1, 0.4]),
    ];

    model.train(&matched_samples, &unmatched_samples);
    
    // Test scoring with all fields missing
    let test_pair = FieldProbabilities::new(vec![f64::NAN, f64::NAN, f64::NAN]);
    let score = model.score(&test_pair);
    
    // Should return 0.0 when no actual comparisons are made
    assert_eq!(score, 0.0);
}

#[test]
fn test_field_probabilities_struct() {
    let probabilities = FieldProbabilities::new(vec![0.9, f64::NAN, 0.7]);
    
    // Test field access
    assert_eq!(probabilities.len(), 3);
    assert!(!probabilities.is_field_missing(0));  // First field is present
    assert!(probabilities.is_field_missing(1));   // Second field is missing
    assert!(!probabilities.is_field_missing(2));  // Third field is present
    
    // Test get_field_probability 
    assert_eq!(probabilities.get_field_probability(0), Some(0.9));
    assert_eq!(probabilities.get_field_probability(1), None);
    assert_eq!(probabilities.get_field_probability(2), Some(0.7));
}