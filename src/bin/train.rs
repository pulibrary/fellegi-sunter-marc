//! Example usage of the Fellegi-Sunter model

use std::{fs::File, io::Write};

use fellegi_sunter_marc::{
    BENCHMARK_MARC, ClusterData, FellegiSunterModel, TRAINING_CLUSTERS, block,
    similarities_between_records,
};
use itertools::Itertools;

const SCORE_THRESHOLD: f64 = 27.0;
const FIELD_COUNT: usize = 12;

fn main() {
    let mut model = FellegiSunterModel::new(FIELD_COUNT);

    let matched_samples = TRAINING_CLUSTERS.clustered_similarities();
    let unmatched_samples = TRAINING_CLUSTERS.unclustered_similarities();

    model.train(&matched_samples, &unmatched_samples);
    println!("==== Done training ====");

    let mut found_clusters: Vec<Vec<&str>> = Vec::default();
    BENCHMARK_MARC.iter().combinations(2).for_each(|pair| {
        if block(pair[0]) == block(pair[1]) {
            let first_title = pair[0].extract_values("245a");
            let second_title = pair[1].extract_values("245a");
            let score = model.score(&similarities_between_records(pair[0], pair[1]));
            if score > SCORE_THRESHOLD {
                println!(
                    "Similarities between fields between {first_title:?} and {second_title:?}: {:?}",
                    similarities_between_records(pair[0], pair[1])
                );
                println!("Score {first_title:?} and {second_title:?}: {score}");
                // println!("Probability match {first_title:?} and {second_title:?}: {probability}");

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
                if (a_id.is_some_and(|id| id == "SCSB-3634720") && b_id.is_some_and(|id| id == "SCSB-10119371")) || a_id.is_some_and(|id| id == "SCSB-10119371") && b_id.is_some_and(|id| id == "SCSB-3634720") {
                    model.weights();
                }
                match (a_id, b_id) {
                    (Some(a), Some(b)) => {
                        found_clusters.push(vec![a, b]);
                    }
                    _ => {}
                }
            }
        }
    });
    println!("P(field|match): {:?}", model.get_p_field_match());
    println!("P(field|non-match): {:?}", model.get_p_field_non_match());
    println!("Field weigths: {:?}", model.weights());
    println!("Prior match: {}", model.get_prior_match());

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
