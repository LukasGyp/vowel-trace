use crate::tracking::tracker::Tracker;
use crate::types::{FormantCandidate, FormantCandidates, FormantPoint};

const DISPLAY_MAX_HZ: f64 = 3500.0;
const SILENCE_FRAMES: usize = 30;

#[derive(Clone, Copy)]
struct Kalman1D {
    x: [f64; 2],
    p: [[f64; 2]; 2],
    initialized: bool,
    residual_count: usize,
    q_boost_remaining: usize,
}

impl Kalman1D {
    fn new() -> Self {
        Self {
            x: [0.0, 0.0],
            p: [[1.0, 0.0], [0.0, 1.0]],
            initialized: false,
            residual_count: 0,
            q_boost_remaining: 0,
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    fn init(&mut self, position: f64) {
        self.x = [position, 0.0];
        self.p = [[1_000.0, 0.0], [0.0, 1_000.0]];
        self.initialized = true;
        self.residual_count = 0;
        self.q_boost_remaining = 0;
    }

    fn predict(&mut self, dt: f64, q_base: f64) {
        if !self.initialized {
            return;
        }
        let q_scale = if self.q_boost_remaining > 0 {
            self.q_boost_remaining -= 1;
            4.0
        } else {
            1.0
        };
        let q = q_base * q_scale;
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt2 * dt2;
        let q11 = q * dt4 / 4.0;
        let q12 = q * dt3 / 2.0;
        let q22 = q * dt2;

        let x0 = self.x[0] + dt * self.x[1];
        let x1 = self.x[1];

        let p00 =
            self.p[0][0] + dt * (self.p[1][0] + self.p[0][1]) + dt2 * self.p[1][1] + q11;
        let p01 = self.p[0][1] + dt * self.p[1][1] + q12;
        let p10 = self.p[1][0] + dt * self.p[1][1] + q12;
        let p11 = self.p[1][1] + q22;

        self.x = [x0, x1];
        self.p = [[p00, p01], [p10, p11]];
    }

    fn update(&mut self, z: f64, r: f64) {
        if !self.initialized {
            self.init(z);
            return;
        }
        let y = z - self.x[0];
        let s = self.p[0][0] + r;
        let k0 = self.p[0][0] / s;
        let k1 = self.p[1][0] / s;

        self.x[0] += k0 * y;
        self.x[1] += k1 * y;

        let p00 = (1.0 - k0) * self.p[0][0];
        let p01 = (1.0 - k0) * self.p[0][1];
        let p10 = self.p[1][0] - k1 * self.p[0][0];
        let p11 = self.p[1][1] - k1 * self.p[0][1];
        self.p = [[p00, p01], [p10, p11]];
    }

    fn position(&self) -> f64 {
        self.x[0]
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn register_residual(&mut self, residual: f64) {
        if residual > 350.0 {
            self.residual_count += 1;
        } else {
            self.residual_count = 0;
        }
        if self.residual_count >= 3 {
            self.q_boost_remaining = 5;
            self.residual_count = 0;
        }
    }
}

pub(crate) struct KalmanTracker {
    f1: Kalman1D,
    f2: Kalman1D,
    silence_frames: usize,
    q_base: f64,
    r_base: f64,
    max_jump_hz: f64,
}

impl KalmanTracker {
    pub(crate) fn new() -> Self {
        Self {
            f1: Kalman1D::new(),
            f2: Kalman1D::new(),
            silence_frames: 0,
            q_base: 2500.0,
            r_base: 2000.0,
            max_jump_hz: 500.0,
        }
    }

    pub(crate) fn set_params(&mut self, q: f32, r: f32, max_jump_hz: f32) {
        self.q_base = q.max(1.0) as f64;
        self.r_base = r.max(1.0) as f64;
        self.max_jump_hz = max_jump_hz.max(50.0) as f64;
    }

    fn pick_smallest_pair(
        &self,
        candidates: &[FormantCandidate],
    ) -> Option<(FormantCandidate, FormantCandidate)> {
        let mut sorted: Vec<FormantCandidate> = candidates
            .iter()
            .copied()
            .filter(|c| c.freq_hz <= DISPLAY_MAX_HZ)
            .collect();
        if sorted.len() < 2 {
            return None;
        }
        sorted.sort_by(|a, b| a.freq_hz.partial_cmp(&b.freq_hz).unwrap());
        Some((sorted[0], sorted[1]))
    }

    fn adjusted_r(&self, strength: f32, voicing_conf: f32, residual: f64) -> f64 {
        let strength_scale = (1.0 - strength as f64).powi(2).clamp(0.5, 4.0);
        let voicing_scale = 1.0 / (voicing_conf.clamp(0.2, 1.0) as f64);
        let mut r = self.r_base * strength_scale * voicing_scale;
        if residual > self.max_jump_hz {
            let extra = (residual / self.max_jump_hz).min(3.0);
            r *= extra * extra;
        }
        r
    }
}

impl Tracker for KalmanTracker {
    fn reset(&mut self, _sample_rate_hz: f32) {
        self.f1.reset();
        self.f2.reset();
        self.silence_frames = 0;
    }

    fn update(&mut self, dt: f32, measurement: Option<FormantCandidates>) -> Option<FormantPoint> {
        let (candidates, voicing_conf, voiced) = match measurement {
            Some(m) => (m.candidates, m.voicing_conf, m.voiced),
            None => (Vec::new(), 0.0, false),
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

        let dt = dt as f64;
        self.f1.predict(dt, self.q_base);
        self.f2.predict(dt, self.q_base);

        let pred_f1 = self.f1.position();
        let pred_f2 = self.f2.position();
        let pred_f1_state = self.f1;
        let pred_f2_state = self.f2;

        let best_pair = self.pick_smallest_pair(&candidates);
        let mut meas_f1: Option<FormantCandidate> = None;
        let mut meas_f2: Option<FormantCandidate> = None;

        if let Some((c1, c2)) = best_pair {
            meas_f1 = Some(c1);
            meas_f2 = Some(c2);
        }

        if let Some(c) = meas_f1 {
            let residual = (c.freq_hz - pred_f1).abs();
            self.f1.register_residual(residual);
            let r = self.adjusted_r(c.strength, voicing_conf, residual);
            self.f1.update(c.freq_hz, r);
        }
        if let Some(c) = meas_f2 {
            let residual = (c.freq_hz - pred_f2).abs();
            self.f2.register_residual(residual);
            let r = self.adjusted_r(c.strength, voicing_conf, residual);
            self.f2.update(c.freq_hz, r);
        }

        let f1 = self.f1.position();
        let f2 = self.f2.position();
        if f1 > f2 {
            self.f1 = pred_f1_state;
            self.f2 = pred_f2_state;
            return None;
        }

        if !self.f1.is_initialized() || !self.f2.is_initialized() {
            return None;
        }

        Some(FormantPoint { t: 0.0, f1, f2 })
    }
}
