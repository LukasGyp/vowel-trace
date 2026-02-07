#[derive(Debug, Clone)]
pub(crate) struct FormantPoint {
    pub(crate) t: f64,
    pub(crate) f1: f64,
    pub(crate) f2: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FormantCandidate {
    pub(crate) freq_hz: f64,
    pub(crate) bandwidth_hz: f64,
    pub(crate) strength: f32, // 0.0 - 1.0
}

#[derive(Clone, Debug)]
pub(crate) struct FormantCandidates {
    pub(crate) candidates: Vec<FormantCandidate>,
    pub(crate) voiced: bool,
    pub(crate) voicing_conf: f32, // 0.0 - 1.0
}

#[derive(Debug, Clone)]
pub(crate) struct SpectrumFrame {
    pub(crate) t: f64,
    pub(crate) spectrum_db: Vec<f32>,
    pub(crate) lpc_env_db: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DebugInfo {
    pub(crate) rms: f32,
    pub(crate) rms_gate: f32,
    pub(crate) voiced: bool,
    pub(crate) peaks: usize,
    pub(crate) formants: usize,
}
