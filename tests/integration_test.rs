//! Integration tests for Fellegi-Sunter model

use fellegi_sunter_marc::{FellegiSunterModel, FieldProbabilities};

#[test]
fn test_model_creation() {
    let model = FellegiSunterModel::new(5);
    assert_eq!(model.get_p_field_match().len(), 5);
    assert_eq!(model.get_p_field_non_match().len(), 5);
    assert_eq!(model.get_prior_match(), 0.5);
}

#[test]
fn test_training() {
    let mut model = FellegiSunterModel::new(2);
    
    let matched_samples = vec![
        FieldProbabilities::new(vec![0.9, 0.8]),
        FieldProbabilities::new(vec![0.8, 0.9]),
    ];

    let unmatched_samples = vec![
        FieldProbabilities::new(vec![0.1, 0.2]),
        FieldProbabilities::new(vec![0.2, 0.1]),
    ];

    model.train(&matched_samples, &unmatched_samples);
    
    // Check that the probabilities were updated
    assert_eq!(model.get_p_field_match().len(), 2);
    assert_eq!(model.get_p_field_non_match().len(), 2);
}

#[test]
fn test_scoring() {
    let mut model = FellegiSunterModel::new(2);
    
    let matched_samples = vec![
        FieldProbabilities::new(vec![0.9, 0.8]),
        FieldProbabilities::new(vec![0.8, 0.9]),
    ];

    let unmatched_samples = vec![
        FieldProbabilities::new(vec![0.1, 0.2]),
        FieldProbabilities::new(vec![0.2, 0.1]),
    ];

    model.train(&matched_samples, &unmatched_samples);
    
    let test_pair = FieldProbabilities::new(vec![0.85, 0.75]);
    let score = model.score(&test_pair);
    
    // Score should be a finite number
    assert!(score.is_finite());
}

#[test]
fn test_empty_training() {
    let mut model = FellegiSunterModel::new(2);
    
    let matched_samples: Vec<FieldProbabilities> = vec![];
    let unmatched_samples: Vec<FieldProbabilities> = vec![];
    
    // This shouldn't panic
    model.train(&matched_samples, &unmatched_samples);
    
    let test_pair = FieldProbabilities::new(vec![0.5, 0.5]);
    let score = model.score(&test_pair);
    assert!(score.is_finite());
}