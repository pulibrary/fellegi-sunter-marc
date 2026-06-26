//! This module is responsible for converting a pair of MARC records into a FieldProbabilities

use std::sync::LazyLock;

use marctk::Record;
use strsim::jaro_winkler;

use crate::FieldProbabilities;

pub fn similarities_between_records(a: &Record, b: &Record) -> FieldProbabilities {
    FieldProbabilities::new(vec![
        subfield_similarity("245abp", a, b),
        subfield_similarity("300a", a, b),
        subfield_similarity("250a", a, b),
        subfield_similarity("260a:264a", a, b),
        subfield_similarity("260b:264b", a, b),
        subfield_similarity("260c:264c", a, b),
        subfield_similarity("245p", a, b),
        subfield_similarity("245n", a, b),
        subfield_similarity("245f", a, b),
        subfield_similarity("100a:110a:111a:130a", a, b),
        subfield_similarity("086", a, b),
    ])
}

fn subfield_similarity(spec: &str, a: &Record, b: &Record) -> f64 {
    match (
        a.extract_values(spec).first().map(|a| normalize(a)),
        b.extract_values(spec).first().map(|b| normalize(b)),
    ) {
        (Some(a), Some(b)) => jaro_winkler(&a, &b),
        _ => f64::NAN,
    }
}

fn normalize(s: &str) -> String {
    s.trim()
        .to_lowercase()
        .chars()
        .filter(|c| !c.is_ascii_punctuation())
        .collect()
}

pub fn get_training_record(id: &str) -> Option<&Record> {
    TRAINING_MARC.iter().find(|r| {
        r.get_control_fields("001")
            .iter()
            .any(|f| f.content() == id)
    })
}

pub static TRAINING_MARC: LazyLock<Vec<Record>> = LazyLock::new(|| {
    Record::from_xml(include_str!("../training-marc.xml"))
        .filter_map(|r| r.ok())
        .collect()
});

pub static BENCHMARK_MARC: LazyLock<Vec<Record>> = LazyLock::new(|| {
    Record::from_xml(include_str!("../b2011.xml"))
        .filter_map(|r| r.ok())
        .collect()
});
