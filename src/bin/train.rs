//! Example usage of the Fellegi-Sunter model

use fellegi_sunter_marc::{
    BENCHMARK_MARC, FellegiSunterModel, TRAINING_CLUSTERS, similarities_between_records,
};
use itertools::Itertools;

fn main() {
    // Create a new model with 3 fields
    let mut model = FellegiSunterModel::new(3);

    // Train the model with some sample data
    let matched_samples = TRAINING_CLUSTERS.clustered_similarities();

    let unmatched_samples = TRAINING_CLUSTERS.unclustered_similarities();

    model.train(&matched_samples, &unmatched_samples);

    let mut found_clusters: Vec<Vec<&str>> = Vec::default();
    BENCHMARK_MARC.iter().combinations(2).for_each(|pair| {
        let first_id = pair[0]
            .get_control_fields("001")
            .first()
            .map(|f| f.content())
            .unwrap();
        let second_id = pair[1]
            .get_control_fields("001")
            .first()
            .map(|f| f.content())
            .unwrap();
        let score = model.score(&similarities_between_records(pair[0], pair[1]));
        if score > -4.0 {
            println!(
                "Similarities between fields between {first_id} and {second_id}: {:?}",
                similarities_between_records(pair[0], pair[1])
            );
            println!("Log likelihood of match between {first_id} and {second_id}: {score}")
        }
    });
}
