//! Train on training data and attempt the benchmark

use std::{fs::File, io::Write};

use fellegi_sunter_marc::{
    BENCHMARK_MARC, ClusterData, FellegiSunterModel, TRAINING_CLUSTERS, block, get_id,
    similarities_between_records,
};
use itertools::Itertools;

// A higher threshold is stricter (fewer matches), lower threshold is more permissive (more matches)
const SCORE_THRESHOLD: f64 = 14.0;
const FIELD_COUNT: usize = 23;

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
            // println!("score {score}");
            if score > SCORE_THRESHOLD {
                println!(
                    "Similarities between fields between {first_title:?} ({:?}) and {second_title:?} ({:?}): {:?}",
                    get_id(pair[0]),
                    get_id(pair[1]),
                    similarities_between_records(pair[0], pair[1])
                );
                println!("Score {first_title:?} and {second_title:?}: {score}");
                // println!("Probability match {first_title:?} and {second_title:?}: {probability}");

                match (get_id(pair[0]), get_id(pair[1])) {
                    (Some(a), Some(b)) => {
                        found_clusters.push(vec![a, b]);
                    }
                    _ => {}
                }
            }
        }
    });
    println!("Field match weigths: {:?}", model.match_weights());
    println!("Field unmatch weigths: {:?}", model.unmatch_weights());
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
