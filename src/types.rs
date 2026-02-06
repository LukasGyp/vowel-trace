#[derive(Debug, Clone)]
pub(crate) struct FormantPoint {
    pub(crate) t: f64,
    pub(crate) f1: f64,
    pub(crate) f2: f64,
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
    pub(crate) voiced: bool,
    pub(crate) peaks: usize,
    pub(crate) formants: usize,
}
