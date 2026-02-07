pub(crate) mod kalman;
pub(crate) mod raw;
pub(crate) mod tracker;
pub(crate) mod viterbi;

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use crate::tracking::tracker::TrackingMode;

#[derive(Debug, Clone, Copy)]
pub(crate) struct TrackingParamsSnapshot {
    pub(crate) mode: TrackingMode,
    pub(crate) kalman_q: f32,
    pub(crate) kalman_r: f32,
    pub(crate) kalman_max_jump_hz: f32,
    pub(crate) viterbi_transition_smoothness: f32,
    pub(crate) viterbi_dropout_penalty: f32,
    pub(crate) viterbi_strength_weight: f32,
}

pub(crate) struct TrackingParams {
    mode: AtomicUsize,
    kalman_q: AtomicU32,
    kalman_r: AtomicU32,
    kalman_max_jump_hz: AtomicU32,
    viterbi_transition_smoothness: AtomicU32,
    viterbi_dropout_penalty: AtomicU32,
    viterbi_strength_weight: AtomicU32,
}

impl TrackingParams {
    pub(crate) fn new() -> Self {
        Self {
            mode: AtomicUsize::new(TrackingMode::Kalman.as_usize()),
            kalman_q: AtomicU32::new(1_000_000.0f32.to_bits()),
            kalman_r: AtomicU32::new(5.0f32.to_bits()),
            kalman_max_jump_hz: AtomicU32::new(500.0f32.to_bits()),
            viterbi_transition_smoothness: AtomicU32::new(0.003f32.to_bits()),
            viterbi_dropout_penalty: AtomicU32::new(3.0f32.to_bits()),
            viterbi_strength_weight: AtomicU32::new(2.0f32.to_bits()),
        }
    }

    pub(crate) fn snapshot(&self) -> TrackingParamsSnapshot {
        TrackingParamsSnapshot {
            mode: TrackingMode::from_usize(self.mode.load(Ordering::Relaxed)),
            kalman_q: f32::from_bits(self.kalman_q.load(Ordering::Relaxed)),
            kalman_r: f32::from_bits(self.kalman_r.load(Ordering::Relaxed)),
            kalman_max_jump_hz: f32::from_bits(self.kalman_max_jump_hz.load(Ordering::Relaxed)),
            viterbi_transition_smoothness: f32::from_bits(
                self.viterbi_transition_smoothness.load(Ordering::Relaxed),
            ),
            viterbi_dropout_penalty: f32::from_bits(
                self.viterbi_dropout_penalty.load(Ordering::Relaxed),
            ),
            viterbi_strength_weight: f32::from_bits(
                self.viterbi_strength_weight.load(Ordering::Relaxed),
            ),
        }
    }

    pub(crate) fn mode(&self) -> TrackingMode {
        TrackingMode::from_usize(self.mode.load(Ordering::Relaxed))
    }

    pub(crate) fn set_mode(&self, mode: TrackingMode) {
        self.mode.store(mode.as_usize(), Ordering::Relaxed);
    }

    pub(crate) fn kalman_q(&self) -> f32 {
        f32::from_bits(self.kalman_q.load(Ordering::Relaxed))
    }

    pub(crate) fn set_kalman_q(&self, value: f32) {
        self.kalman_q.store(value.to_bits(), Ordering::Relaxed);
    }

    pub(crate) fn kalman_r(&self) -> f32 {
        f32::from_bits(self.kalman_r.load(Ordering::Relaxed))
    }

    pub(crate) fn set_kalman_r(&self, value: f32) {
        self.kalman_r.store(value.to_bits(), Ordering::Relaxed);
    }

    pub(crate) fn kalman_max_jump_hz(&self) -> f32 {
        f32::from_bits(self.kalman_max_jump_hz.load(Ordering::Relaxed))
    }

    pub(crate) fn set_kalman_max_jump_hz(&self, value: f32) {
        self.kalman_max_jump_hz
            .store(value.to_bits(), Ordering::Relaxed);
    }

    pub(crate) fn viterbi_transition_smoothness(&self) -> f32 {
        f32::from_bits(self.viterbi_transition_smoothness.load(Ordering::Relaxed))
    }

    pub(crate) fn set_viterbi_transition_smoothness(&self, value: f32) {
        self.viterbi_transition_smoothness
            .store(value.to_bits(), Ordering::Relaxed);
    }

    pub(crate) fn viterbi_dropout_penalty(&self) -> f32 {
        f32::from_bits(self.viterbi_dropout_penalty.load(Ordering::Relaxed))
    }

    pub(crate) fn set_viterbi_dropout_penalty(&self, value: f32) {
        self.viterbi_dropout_penalty
            .store(value.to_bits(), Ordering::Relaxed);
    }

    pub(crate) fn viterbi_strength_weight(&self) -> f32 {
        f32::from_bits(self.viterbi_strength_weight.load(Ordering::Relaxed))
    }

    pub(crate) fn set_viterbi_strength_weight(&self, value: f32) {
        self.viterbi_strength_weight
            .store(value.to_bits(), Ordering::Relaxed);
    }
}

impl Default for TrackingParams {
    fn default() -> Self {
        Self::new()
    }
}
