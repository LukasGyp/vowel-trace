use std::collections::VecDeque;

use crate::tracking::tracker::Tracker;
use crate::types::{FormantCandidate, FormantCandidates, FormantPoint};

const DISPLAY_MAX_HZ: f64 = 3500.0;
const SILENCE_FRAMES: usize = 30;
const WINDOW_LEN: usize = 30;

#[derive(Clone, Copy)]
struct State {
    f1: Option<FormantCandidate>,
    f2: Option<FormantCandidate>,
    emission: f64,
}

struct FrameCandidates {
    candidates: Vec<FormantCandidate>,
    voicing_conf: f32,
}

pub(crate) struct ViterbiTracker {
    window: VecDeque<FrameCandidates>,
    silence_frames: usize,
    w_t: f64,
    w_d: f64,
    w_s: f64,
}

impl ViterbiTracker {
    pub(crate) fn new() -> Self {
        Self {
            window: VecDeque::new(),
            silence_frames: 0,
            w_t: 0.003,
            w_d: 3.0,
            w_s: 2.0,
        }
    }

    pub(crate) fn set_params(
        &mut self,
        transition_smoothness: f32,
        dropout_penalty: f32,
        strength_weight: f32,
    ) {
        self.w_t = transition_smoothness.max(0.0) as f64;
        self.w_d = dropout_penalty.max(0.0) as f64;
        self.w_s = strength_weight.max(0.0) as f64;
    }

    fn push_frame(&mut self, measurement: Option<FormantCandidates>) {
        let (candidates, voicing_conf) = match measurement {
            Some(m) => (m.candidates, m.voicing_conf),
            None => (Vec::new(), 0.0),
        };

        let mut filtered: Vec<FormantCandidate> = candidates
            .into_iter()
            .filter(|c| c.freq_hz <= DISPLAY_MAX_HZ)
            .collect();
        filtered.sort_by(|a, b| a.freq_hz.partial_cmp(&b.freq_hz).unwrap());

        self.window.push_back(FrameCandidates {
            candidates: filtered,
            voicing_conf,
        });
        while self.window.len() > WINDOW_LEN {
            self.window.pop_front();
        }
    }

    fn emission_cost(&self, c: FormantCandidate, voicing_conf: f32) -> f64 {
        let bw_penalty = if c.bandwidth_hz <= 400.0 {
            0.0
        } else {
            (c.bandwidth_hz - 400.0) / 400.0
        };
        let conf_scale = 1.0 + (1.0 - voicing_conf as f64);
        (self.w_s * conf_scale) * (1.0 - c.strength as f64) + conf_scale * bw_penalty
    }

    fn build_states(&self, frame: &FrameCandidates) -> Vec<State> {
        let mut states = Vec::new();
        if frame.candidates.len() >= 2 {
            let c1 = frame.candidates[0];
            let c2 = frame.candidates[1];
            let emission = self.emission_cost(c1, frame.voicing_conf)
                + self.emission_cost(c2, frame.voicing_conf);
            states.push(State {
                f1: Some(c1),
                f2: Some(c2),
                emission,
            });
        }
        if states.is_empty() {
            states.push(State {
                f1: None,
                f2: None,
                emission: self.w_d,
            });
        }
        states
    }

    fn transition_cost(&self, prev: &State, next: &State) -> f64 {
        match (prev.f1, prev.f2, next.f1, next.f2) {
            (Some(p1), Some(p2), Some(n1), Some(n2)) => {
                self.w_t * ((n1.freq_hz - p1.freq_hz).abs() + (n2.freq_hz - p2.freq_hz).abs())
            }
            _ => 0.0,
        }
    }

    fn run_viterbi(&self) -> Option<FormantPoint> {
        if self.window.is_empty() {
            return None;
        }

        let mut dp: Vec<f64> = Vec::new();
        let mut states_prev: Vec<State> = Vec::new();

        for (frame_idx, frame) in self.window.iter().enumerate() {
            let states = self.build_states(frame);
            if frame_idx == 0 {
                dp = states.iter().map(|s| s.emission).collect();
                states_prev = states;
                continue;
            }

            let mut dp_curr = vec![f64::INFINITY; states.len()];
            for (j, state) in states.iter().enumerate() {
                let mut best = f64::INFINITY;
                for (i, prev_state) in states_prev.iter().enumerate() {
                    let cost = dp[i] + self.transition_cost(prev_state, state);
                    if cost < best {
                        best = cost;
                    }
                }
                dp_curr[j] = best + state.emission;
            }
            dp = dp_curr;
            states_prev = states;
        }

        let (best_idx, _) = dp
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())?;
        let best_state = &states_prev[best_idx];
        let (Some(f1), Some(f2)) = (best_state.f1, best_state.f2) else {
            return None;
        };

        Some(FormantPoint {
            t: 0.0,
            f1: f1.freq_hz,
            f2: f2.freq_hz,
        })
    }
}

impl Tracker for ViterbiTracker {
    fn reset(&mut self, _sample_rate_hz: f32) {
        self.window.clear();
        self.silence_frames = 0;
    }

    fn update(&mut self, _dt: f32, measurement: Option<FormantCandidates>) -> Option<FormantPoint> {
        let voiced = match measurement.as_ref() {
            Some(m) => m.voiced,
            None => false,
        };
        if !voiced {
            self.silence_frames += 1;
            if self.silence_frames >= SILENCE_FRAMES {
                self.reset(0.0);
                return None;
            }
        } else {
            self.silence_frames = 0;
        }

        self.push_frame(measurement);

        self.run_viterbi()
    }
}
