//! This module is responsible for converting a pair of MARC records into a FieldProbabilities

use core::f64;
use std::sync::LazyLock;

use itertools::Itertools;
use marctk::Record;
use strsim::{jaro_winkler, sorensen_dice};

use crate::FieldProbabilities;

pub fn similarities_between_records(a: &Record, b: &Record) -> FieldProbabilities {
    FieldProbabilities::new(vec![
        fuzzy_subfield_similarity("100a:110a:111a:130a", a, b),
        fuzzy_subfield_similarity("245abfnp", a, b),
        fuzzy_subfield_similarity("260a:264a", a, b),
        fuzzy_subfield_similarity("260b:264b", a, b),
        // fuzzy_numeric_match("086", a, b),
        fuzzy_numeric_match("250a", a, b),
        fuzzy_numeric_match("300c", a, b),
        exact_oclc_number_match(a, b),
        exact_number_match("010a", a, b),
        exact_number_match("260c:264c", a, b),
        exact_number_match("300a", a, b),
        year_from_008_similarity(a, b),
    ])
}

fn year_from_008_similarity(a: &Record, b: &Record) -> f64 {
    match (
        a.get_control_fields("008")
            .first()
            .and_then(|f| f.content().get(7..11)),
        b.get_control_fields("008")
            .first()
            .and_then(|f| f.content().get(7..11)),
    ) {
        (Some(a), Some(b)) => exponential_numeric_difference(a, b),
        _ => 0.0,
    }
}

fn year_from_008_exact_match(a: &Record, b: &Record) -> f64 {
    match (
        a.get_control_fields("008")
            .first()
            .and_then(|f| f.content().get(7..11)),
        b.get_control_fields("008")
            .first()
            .and_then(|f| f.content().get(7..11)),
    ) {
        (Some(a), Some(b)) if a == b => 1.0,
        (Some(_), Some(_)) => 0.0,
        _ => 0.0,
    }
}

fn exact_subfield_match(spec: &str, a: &Record, b: &Record) -> f64 {
    match (
        a.extract_values(spec).first().map(|a| normalize(a)),
        b.extract_values(spec).first().map(|b| normalize(b)),
    ) {
        (Some(a), Some(b)) if a == b => 1.0,
        (Some(_), Some(_)) => 0.0,
        _ => 0.0,
    }
}

fn exact_oclc_number_match(a: &Record, b: &Record) -> f64 {
    match (
        a.extract_values("035a").first(),
        b.extract_values("035a").first(),
    ) {
        (Some(a), Some(b)) => {
            if (a.starts_with("(OCoLC)") || a.starts_with("ocm") || a.starts_with("ocn"))
                && (b.starts_with("(OCoLC)") || b.starts_with("ocm") || b.starts_with("ocn"))
            {
                if a.chars()
                    .filter(|c| c.is_numeric())
                    .eq(b.chars().filter(|c| c.is_numeric()))
                {
                    1.0
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

fn exact_number_match(spec: &str, a: &Record, b: &Record) -> f64 {
    match (
        a.extract_values(spec).first(),
        b.extract_values(spec).first(),
    ) {
        (Some(a), Some(b)) => {
            if normalize_numeric(a) == normalize_numeric(b) {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

fn fuzzy_numeric_match(spec: &str, a: &Record, b: &Record) -> f64 {
    match (
        a.extract_values(spec).first(),
        b.extract_values(spec).first(),
    ) {
        (Some(a), Some(b)) => exponential_numeric_difference(a, b),
        _ => 0.0,
    }
}

fn exponential_numeric_difference(a: &str, b: &str) -> f64 {
    let parsed_a = normalize_numeric(a).parse::<usize>();
    let parsed_b = normalize_numeric(b).parse::<usize>();
    match (parsed_a, parsed_b) {
        (Ok(pa), Ok(pb)) => f64::consts::E.powi(pa.abs_diff(pb) as i32 * -1),
        _ => 0.0,
    }
}

fn fuzzy_subfield_similarity(spec: &str, a: &Record, b: &Record) -> f64 {
    match (
        a.extract_values(spec).first().map(|a| normalize(a)),
        b.extract_values(spec).first().map(|b| normalize(b)),
    ) {
        (Some(a), Some(b)) => jaro_winkler(&a, &b),
        _ => 0.0,
    }
}

fn sorensen_dice_similarity(spec: &str, a: &Record, b: &Record) -> f64 {
    match (
        a.extract_values(spec).first().map(|a| normalize(a)),
        b.extract_values(spec).first().map(|b| normalize(b)),
    ) {
        (Some(a), Some(b)) => sorensen_dice(&a, &b),
        _ => 0.0,
    }
}

fn normalize(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .filter(|w| !stop_words::get(stop_words::LANGUAGE::English).contains(w))
        .join(" ")
        .chars()
        .filter(|c| !c.is_ascii_punctuation())
        .collect::<String>()
}

fn normalize_numeric(s: &str) -> String {
    s.chars()
        .take_while(|c| c != &',')
        .filter(|c| c.is_numeric())
        .collect()
}

pub fn block(record: &Record) -> [String; 3] {
    [
        // Block on title (without English stopwords)
        record
            .extract_values("245a")
            .first()
            .map(|title| normalize(title))
            .unwrap_or_default(),
        // Block on second and third digits (century and decade) of Date1
        record
            .get_control_fields("008")
            .first()
            .and_then(|f| f.content().get(8..10).map(|s| s.to_string()))
            .unwrap_or_default(),
        // Block on 008 language
        record
            .get_control_fields("008")
            .first()
            .and_then(|f| f.content().get(35..38).map(|s| s.to_string()))
            .unwrap_or_default(),
    ]
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fuzzy_numeric_match() {
        let a = Record::from_breaker("=264 \\$cc2025").unwrap();
        let b = Record::from_breaker("=264 \\$c2025").unwrap();
        let c = Record::from_breaker("=264 \\$c 2025   ").unwrap();
        let d = Record::from_breaker("=264 \\$cc2026").unwrap();
        let e = Record::from_breaker("=264 \\$cc2024").unwrap();
        let f = Record::from_breaker("=264 \\$cc2020").unwrap();

        assert_eq!(fuzzy_numeric_match("264c", &a, &b), 1.0);
        assert_eq!(fuzzy_numeric_match("264c", &a, &c), 1.0);
        assert_eq!(
            fuzzy_numeric_match("264c", &a, &d),
            fuzzy_numeric_match("264c", &a, &e)
        );
        assert!(
            fuzzy_numeric_match("264c", &a, &d) < 0.5,
            "the similarity gets dramatically lower when not an exact match"
        );
        assert!(fuzzy_numeric_match("264c", &a, &f) < 0.01);
    }

    #[test]
    fn test_it_puts_different_decades_in_different_blocks() {
        let a = Record::from_breaker("=008 731224s1972    ctua     bs   001 0 eng").unwrap();
        let b = Record::from_breaker("=008 730424s1964    ctua     b    000 0 eng").unwrap();
        assert!(block(&a) != block(&b))
    }
}
