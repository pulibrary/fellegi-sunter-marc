//! Train on training data and attempt the benchmark

use std::{fs::File, io::Write};

use fellegi_sunter_marc::{
    BENCHMARK_MARC, ClusterData, FellegiSunterModel, TRAINING_CLUSTERS, block, get_id, pub_year,
    similarities_between_records,
};
use itertools::Itertools;

// A higher threshold is stricter (fewer matches), lower threshold is more permissive (more matches)
const SCORE_THRESHOLD: f64 = 12.0;
const FIELD_COUNT: usize = 22;

fn main() {
    let mut model = FellegiSunterModel::new(FIELD_COUNT);

    let matched_samples = TRAINING_CLUSTERS.clustered_similarities();
    let unmatched_samples = TRAINING_CLUSTERS.unclustered_similarities();

    model.train(&matched_samples, &unmatched_samples);
    println!("==== Done training ====");

    let mut csv_report = File::create("pairs_report.csv").unwrap();
    writeln!(&mut csv_report, "score,id1,title1,year1,pagination1,edition1,publisher1,type1,id2,title2,year2,pagination2,edition2,publisher2,type2,\"Weighted 1xx\",\"Weighted 245\",\"Weighted publisher\",\"Weighted edition\",\"Weighted 086\",\"Weighted 300b\",\"Weighted 300c\",\"Weighted OCLC\",\"Weighted LCCN\",\"Weighted ISBN\",\"Weighted 050\",\"Weighted series number\",\"Weighted pagination\",\"Weighted pub year\",\"Weighted 500\",\"Weighted 505\",\"Weighted 040\",\"Weighted 042\",\"Weighted record creation date\",\"Weighted 008/15-17\",\"Weighted LDR/07\",\"Weighted SCSB partner library match\"").unwrap();

    let mut found_clusters: Vec<Vec<&str>> = Vec::default();

    let match_weights = model.match_weights();
    BENCHMARK_MARC.iter().combinations(2).for_each(|pair| {
        if block(pair[0]) == block(pair[1]) {
            let first_title = pair[0].extract_values("245a");
            let second_title = pair[1].extract_values("245a");
            let score = model.score(&similarities_between_records(pair[0], pair[1]));
            // println!("score {score}");
            if score > SCORE_THRESHOLD {
                let similarities = similarities_between_records(pair[0], pair[1]);
                println!(
                    "Similarities between fields between {first_title:?} ({:?}) and {second_title:?} ({:?}): {:?}",
                    get_id(pair[0]),
                    get_id(pair[1]),
                    similarities
                );
                println!("Score {first_title:?} and {second_title:?}: {score}");
                // println!("Probability match {first_title:?} and {second_title:?}: {probability}");

                match (get_id(pair[0]), get_id(pair[1])) {
                    (Some(a), Some(b)) => {
                        found_clusters.push(vec![a, b]);
                    }
                    _ => {}
                }

                writeln!(&mut csv_report, "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"",
                    score,
                    get_id(pair[0]).unwrap_or_default().replace('"', "\'"),
                    first_title.first().unwrap_or(&"").replace('"', "\'"),
                    pub_year(pair[0]).unwrap_or_default().replace('"', "\'"),
                    pair[0].extract_values("300a").first().unwrap_or(&"").replace('"', "\'"),
                    pair[0].extract_values("250a").first().unwrap_or(&"").replace('"', "\'"),
                    pair[0].extract_values("260a:264a").first().unwrap_or(&"").replace('"', "\'"),
                    pair[0].leader().chars().nth(6).unwrap_or_default(),
                    get_id(pair[1]).unwrap_or_default().replace('"', "\'"),
                    second_title.first().unwrap_or(&"").replace('"', "\'"),
                    pub_year(pair[1]).unwrap_or_default().replace('"', "\'"),
                    pair[1].extract_values("300a").first().unwrap_or(&"").replace('"', "\'"),
                    pair[1].extract_values("250a").first().unwrap_or(&"").replace('"', "\'"),
                    pair[1].extract_values("260a:264a").first().unwrap_or(&"").replace('"', "\'"),
                    pair[1].leader().chars().nth(6).unwrap_or_default(),
                    similarities.probabilities[0] * match_weights[0],
                    similarities.probabilities[1] * match_weights[1],
                    similarities.probabilities[2] * match_weights[2],
                    similarities.probabilities[3] * match_weights[3],
                    similarities.probabilities[4] * match_weights[4],
                    similarities.probabilities[5] * match_weights[5],
                    similarities.probabilities[6] * match_weights[6],
                    similarities.probabilities[7] * match_weights[7],
                    similarities.probabilities[8] * match_weights[8],
                    similarities.probabilities[9] * match_weights[9],
                    similarities.probabilities[10] * match_weights[10],
                    similarities.probabilities[11] * match_weights[11],
                    similarities.probabilities[12] * match_weights[12],
                    similarities.probabilities[13] * match_weights[13],
                    similarities.probabilities[14] * match_weights[14],
                    similarities.probabilities[15] * match_weights[15],
                    similarities.probabilities[16] * match_weights[16],
                    similarities.probabilities[17] * match_weights[17],
                    similarities.probabilities[18] * match_weights[18],
                    similarities.probabilities[19] * match_weights[19],
                    similarities.probabilities[20] * match_weights[20],
                    similarities.probabilities[21] * match_weights[21],
                ).unwrap();
            }
        }
    });
    println!("Field match weigths: {:?}", match_weights);
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
