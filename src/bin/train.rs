//! Example usage of the Fellegi-Sunter model

use std::{fs::File, io::Write};

use fellegi_sunter_marc::{
    BENCHMARK_MARC, ClusterData, FellegiSunterModel, TRAINING_CLUSTERS,
    similarities_between_records,
};
use itertools::Itertools;

fn main() {
    // Create a new model with 3 fields
    let mut model = FellegiSunterModel::new(11);

    // Train the model with some sample data
    let matched_samples = TRAINING_CLUSTERS.clustered_similarities();

    let unmatched_samples = TRAINING_CLUSTERS.unclustered_similarities();

    model.train(&matched_samples, &unmatched_samples);

    let mut found_clusters: Vec<Vec<&str>> = Vec::default();
    BENCHMARK_MARC.iter().combinations(2).for_each(|pair| {
        let first_title = pair[0].extract_values("245a");
        let second_title = pair[1].extract_values("245a");
        let score = model.score(&similarities_between_records(pair[0], pair[1]));
        let probability = score.exp2() / (1.0 + score.exp2());
        if probability > 0.999 {
            // println!(
            //     "Similarities between fields between {first_title:?} and {second_title:?}: {:?}",
            //     similarities_between_records(pair[0], pair[1])
            // );
            println!("Probability match {first_title:?} and {second_title:?}: {probability}");
            // println!("P(field|match): {:?}", model.get_p_field_match());
            // println!("P(field|non-match): {:?}", model.get_p_field_non_match());
            // println!("Prior match: {}", model.get_prior_match());

            let a_id = pair[0]
                .get_control_fields("001")
                .iter()
                .map(|f| f.content())
                .next();
            let b_id = pair[1]
                .get_control_fields("001")
                .iter()
                .map(|f| f.content())
                .next();
            match (a_id, b_id) {
                (Some(a), Some(b)) => {
                    found_clusters.push(vec![a, b]);
                }
                _ => {}
            }
        }
    });

    let mut file = File::create("output.json").unwrap();
    file.write_all(
        serde_json::to_string_pretty(&ClusterData {
            clusters: found_clusters,
            unclustered: Vec::default(),
        })
        .unwrap()
        .as_bytes(),
    )
    .unwrap();
}
