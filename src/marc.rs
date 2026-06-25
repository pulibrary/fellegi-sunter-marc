//! This module is responsible for converting a pair of MARC records into a FieldProbabilities

use std::sync::LazyLock;

use marctk::Record;
use strsim::jaro_winkler;

use crate::FieldProbabilities;

pub fn similarities_between_records(a: &Record, b: &Record) -> FieldProbabilities {
    FieldProbabilities::new(vec![
        subfield_similarity("245a", a, b),
        subfield_similarity("300a", a, b),
        subfield_similarity("250a", a, b),
    ])
}

fn subfield_similarity(spec: &str, a: &Record, b: &Record) -> f64 {
    match (
        a.extract_values(spec).first(),
        b.extract_values(spec).first(),
    ) {
        (Some(a), Some(b)) => jaro_winkler(a, b),
        _ => f64::NAN,
    }
}

pub fn get_record(id: &str) -> Option<&Record> {
    TRAINING_MARC.iter().find(|r| {
        r.get_control_fields("001")
            .iter()
            .any(|f| f.content() == id)
    })
}

static TRAINING_MARC: LazyLock<Vec<Record>> = LazyLock::new(|| {
    Record::from_xml(include_str!("../training-marc.xml"))
        .filter_map(|r| r.ok())
        .collect()
});

pub static BENCHMARK_MARC: LazyLock<Vec<Record>> = LazyLock::new(|| {
    Record::from_xml(include_str!("../b2011.xml"))
        .filter_map(|r| r.ok())
        .collect()
});
