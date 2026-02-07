use crate::types::{FormantCandidates, FormantPoint};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrackingMode {
    Raw,
    Kalman,
    Viterbi,
}

impl TrackingMode {
    pub(crate) fn as_usize(self) -> usize {
        match self {
            TrackingMode::Raw => 0,
            TrackingMode::Kalman => 1,
            TrackingMode::Viterbi => 2,
        }
    }

    pub(crate) fn from_usize(value: usize) -> Self {
        match value {
            0 => TrackingMode::Raw,
            2 => TrackingMode::Viterbi,
            _ => TrackingMode::Kalman,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            TrackingMode::Raw => "Raw",
            TrackingMode::Kalman => "Kalman",
            TrackingMode::Viterbi => "Viterbi (experimental)",
        }
    }
}

pub(crate) trait Tracker {
    fn reset(&mut self, sample_rate_hz: f32);
    fn update(&mut self, dt: f32, measurement: Option<FormantCandidates>) -> Option<FormantPoint>;
}

pub(crate) fn candidates_in_range(
    candidates: &[crate::types::FormantCandidate],
    min_hz: f64,
    max_hz: f64,
) -> Vec<crate::types::FormantCandidate> {
    candidates
        .iter()
        .copied()
        .filter(|c| c.freq_hz >= min_hz && c.freq_hz <= max_hz)
        .collect()
}

pub(crate) fn sort_by_strength_desc(candidates: &mut [crate::types::FormantCandidate]) {
    candidates.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());
}
