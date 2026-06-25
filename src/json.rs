use std::sync::LazyLock;

use crate::{
    FieldProbabilities,
    marc::{get_record, similarities_between_records},
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct ClusterData<'a> {
    #[serde(borrow)]
    clusters: Vec<Vec<&'a str>>,
    unclustered: Vec<&'a str>,
}

impl<'a> ClusterData<'a> {
    pub fn unclustered_similarities(&self) -> Vec<FieldProbabilities> {
        let pairs = self.unclustered.iter().combinations(2);
        pairs
            .filter_map(|pair| {
                let record_a = get_record(pair[0])?;
                let record_b = get_record(pair[1])?;
                Some(similarities_between_records(record_a, record_b))
            })
            .collect()
    }

    pub fn clustered_similarities(&self) -> Vec<FieldProbabilities> {
        let pairs = self
            .clusters
            .iter()
            .map(|cluster| cluster.iter().combinations(2))
            .flatten();
        pairs
            .filter_map(|pair| {
                let record_a = get_record(pair[0])?;
                let record_b = get_record(pair[1])?;
                Some(similarities_between_records(record_a, record_b))
            })
            .collect()
    }
}

const TRAINING_CLUSTERS_RAW: &str = include_str!("../training_clusters.json");

pub static TRAINING_CLUSTERS: LazyLock<ClusterData> =
    LazyLock::new(|| serde_json::from_str(TRAINING_CLUSTERS_RAW).unwrap());
