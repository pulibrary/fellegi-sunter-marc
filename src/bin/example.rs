//! Example usage of the Fellegi-Sunter model

use fellegi_sunter_marc::{FellegiSunterModel, FieldProbabilities};

fn main() {
    // Create a new model with 3 fields
    let mut model = FellegiSunterModel::new(3);

    // Train the model with some sample data
    let matched_samples = vec![
        FieldProbabilities::new(vec![0.9, 0.8, 0.7]),
        FieldProbabilities::new(vec![0.8, 0.9, 0.6]),
    ];

    let unmatched_samples = vec![
        FieldProbabilities::new(vec![0.1, 0.2, 0.3]),
        FieldProbabilities::new(vec![0.2, 0.1, 0.4]),
    ];

    model.train(&matched_samples, &unmatched_samples);

    // Run the model on new data
    let test_pair = FieldProbabilities::new(vec![0.85, 0.75, 0.65]);
    let score = model.score(&test_pair);
    println!("Linkage score: {}", score);
    
    println!("P(field|match): {:?}", model.get_p_field_match());
    println!("P(field|non-match): {:?}", model.get_p_field_non_match());
    println!("Prior match: {}", model.get_prior_match());
}