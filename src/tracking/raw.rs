use crate::tracking::tracker::Tracker;
use crate::types::{FormantCandidates, FormantPoint};

const DISPLAY_MAX_HZ: f64 = 3000.0;

pub(crate) struct RawTracker {
}

impl RawTracker {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl Tracker for RawTracker {
    fn reset(&mut self, _sample_rate_hz: f32) {
    }

    fn update(&mut self, _dt: f32, measurement: Option<FormantCandidates>) -> Option<FormantPoint> {
        let measurement = measurement?;
        if !measurement.voiced {
            return None;
        }

        let mut combined: Vec<_> = measurement
            .candidates
            .iter()
            .copied()
            .filter(|c| c.freq_hz <= DISPLAY_MAX_HZ)
            .collect();

        if combined.len() < 2 {
            return None;
        }

        combined.sort_by(|a, b| a.freq_hz.partial_cmp(&b.freq_hz).unwrap());
        let selected = &combined[..2];

        Some(FormantPoint {
            t: 0.0,
            f1: selected[0].freq_hz,
            f2: selected[1].freq_hz,
        })
    }
}
