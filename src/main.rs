use std::collections::VecDeque;
use std::f64::consts::PI;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};
use crossbeam_channel::{Receiver, Sender};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use num_complex::Complex64;
use ringbuf::{HeapConsumer, HeapProducer, HeapRb};
use rustfft::{num_complex::Complex, Fft, FftPlanner};

#[derive(Debug, Clone)]
struct FormantPoint {
    t: f64,
    f1: f64,
    f2: f64,
}

#[derive(Debug, Clone)]
struct SpectrumFrame {
    t: f64,
    spectrum_db: Vec<f32>,
    lpc_env_db: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
struct DebugInfo {
    rms: f32,
    voiced: bool,
    peaks: usize,
    formants: usize,
}

struct FormantApp {
    rx: Receiver<FormantPoint>,
    level_rx: Receiver<f32>,
    voiced_rx: Receiver<bool>,
    debug_rx: Receiver<DebugInfo>,
    spec_rx: Receiver<SpectrumFrame>,
    points: Vec<FormantPoint>,
    last_spectrum: Vec<f32>,
    last_lpc_env: Vec<f32>,
    start: Instant,
    sample_rate: u32,
    spec_bins: usize,
    spec_bin_hz: f64,
    _stream: cpal::Stream,
    status: String,
    last_rms: f32,
    last_detect: Option<f64>,
    last_voiced: bool,
    last_debug: Option<DebugInfo>,
}

type Complex32 = Complex<f32>;

impl FormantApp {
    fn new() -> anyhow::Result<Self> {
        let (rx, level_rx, voiced_rx, debug_rx, spec_rx, sample_rate, stream, cfg) =
            setup_audio()?;
        Ok(Self {
            rx,
            level_rx,
            voiced_rx,
            debug_rx,
            spec_rx,
            points: Vec::new(),
            last_spectrum: Vec::new(),
            last_lpc_env: Vec::new(),
            start: Instant::now(),
            sample_rate,
            spec_bins: cfg.spec_bins,
            spec_bin_hz: cfg.spec_bin_hz,
            _stream: stream,
            status: "running".to_string(),
            last_rms: 0.0,
            last_detect: None,
            last_voiced: false,
            last_debug: None,
        })
    }

    fn spectrum_line(&self, name: &str, data: &[f32]) -> Line {
        let points: PlotPoints = data
            .iter()
            .enumerate()
            .map(|(i, &db)| {
                let freq = (i as f64) * self.spec_bin_hz;
                [freq, db as f64]
            })
            .collect();
        Line::new(points).name(name)
    }

    fn formant_line(&self, name: &str, accessor: impl Fn(&FormantPoint) -> f64) -> Line {
        let mut pts: Vec<[f64; 2]> = Vec::with_capacity(self.points.len() + 8);
        let gap = 0.3;
        let mut last_t: Option<f64> = None;
        for p in &self.points {
            if let Some(prev) = last_t {
                if p.t - prev > gap {
                    pts.push([f64::NAN, f64::NAN]);
                }
            }
            pts.push([p.t, accessor(p)]);
            last_t = Some(p.t);
        }
        Line::new(PlotPoints::from(pts)).name(name)
    }
}

impl eframe::App for FormantApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        for p in self.rx.try_iter() {
            self.last_detect = Some(p.t);
            self.points.push(p);
        }
        for rms in self.level_rx.try_iter() {
            self.last_rms = rms;
        }
        for v in self.voiced_rx.try_iter() {
            self.last_voiced = v;
        }
        for d in self.debug_rx.try_iter() {
            self.last_debug = Some(d);
        }
        let mut pending_spec: Vec<SpectrumFrame> = self.spec_rx.try_iter().collect();
        for mut s in pending_spec.drain(..) {
            self.last_lpc_env = std::mem::take(&mut s.lpc_env_db);
            let column = std::mem::take(&mut s.spectrum_db);
            self.last_spectrum = column;
        }

        let now = self.start.elapsed().as_secs_f64();
        let window_sec = 5.0;
        self.points.retain(|p| now - p.t <= window_sec);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Real-time Formant Tracker");
            ui.label(format!(
                "sample_rate: {} Hz (proc {} Hz)",
                self.sample_rate,
                ProcessingConfig::PROC_SAMPLE_RATE
            ));
            ui.label(format!("status: {}", self.status));
            ui.label(format!("rms: {:.4}", self.last_rms));
            ui.label(format!(
                "voiced: {} (threshold {:.4})",
                if self.last_voiced { "yes" } else { "no" },
                ProcessingConfig::rms_gate_value()
            ));
            if let Some(d) = self.last_debug {
                ui.label(format!(
                    "debug: peaks {}, formants {}",
                    d.peaks, d.formants
                ));
            } else {
                ui.label("debug: none");
            }
            ui.label(format!(
                "params: win 25ms, hop 10ms, max_formant {}Hz, preemph {}, lpc_order {}",
                ProcessingConfig::MAX_FORMANT_HZ,
                ProcessingConfig::PREEMPH_COEF,
                ProcessingConfig::LPC_ORDER
            ));
            if let Some(t) = self.last_detect {
                let delta = now - t;
                ui.label(format!("last detect: {:.2} s ago", delta));
            } else {
                ui.label("last detect: none");
            }

            let f1 = self.formant_line("F1", |p| p.f1);
            let f2 = self.formant_line("F2", |p| p.f2);

            ui.columns(2, |cols| {
                let (left, right) = cols.split_at_mut(1);
                let left = &mut left[0];
                let right = &mut right[0];

                Plot::new("formants")
                    .legend(egui_plot::Legend::default())
                    .height(400.0)
                    .include_y(0.0)
                    .include_y(ProcessingConfig::DISPLAY_MAX_HZ)
                    .include_x(now - window_sec)
                    .include_x(now)
                    .show(left, |plot_ui| {
                        plot_ui.line(f1);
                        plot_ui.line(f2);
                    });

                let spectrogram_plot = Plot::new("spectrogram")
                    .legend(egui_plot::Legend::default())
                    .height(400.0)
                    .include_x(0.0)
                    .include_x(ProcessingConfig::DISPLAY_MAX_HZ)
                    .include_y(ProcessingConfig::SPECTRUM_DB_MIN as f64)
                    .include_y(ProcessingConfig::SPECTRUM_DB_MAX as f64);

                spectrogram_plot.show(right, |plot_ui| {
                    if !self.last_spectrum.is_empty() {
                        plot_ui.line(self.spectrum_line("spectrum", &self.last_spectrum));
                    }
                    if !self.last_lpc_env.is_empty() {
                        plot_ui.line(self.spectrum_line("LPC envelope", &self.last_lpc_env));
                    }
                });
            });
        });

        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

fn setup_audio() -> anyhow::Result<(
    Receiver<FormantPoint>,
    Receiver<f32>,
    Receiver<bool>,
    Receiver<DebugInfo>,
    Receiver<SpectrumFrame>,
    u32,
    cpal::Stream,
    ProcessingConfig,
)> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no input device available"))?;
    let config = choose_input_config(&device, 48_000)?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    let capacity = (sample_rate as usize).saturating_mul(2);
    let rb = HeapRb::<f32>::new(capacity);
    let (producer, consumer) = rb.split();

    let (tx, rx) = crossbeam_channel::unbounded();
    let (level_tx, level_rx) = crossbeam_channel::unbounded();
    let (voiced_tx, voiced_rx) = crossbeam_channel::unbounded();
    let (debug_tx, debug_rx) = crossbeam_channel::unbounded();
    let (spec_tx, spec_rx) = crossbeam_channel::unbounded();
    let start = Instant::now();

    let stream_config: cpal::StreamConfig = config.clone().into();
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => build_stream::<f32>,
        cpal::SampleFormat::I16 => build_stream::<i16>,
        cpal::SampleFormat::U16 => build_stream::<u16>,
        _ => return Err(anyhow::anyhow!("unsupported sample format")),
    }(
        &device,
        &stream_config,
        channels,
        producer,
        stream_error,
    )?;

    let processing = ProcessingConfig::new(sample_rate);
    spawn_processing_thread(
        consumer,
        tx,
        level_tx,
        voiced_tx,
        debug_tx,
        spec_tx,
        start,
        processing.clone(),
    );

    stream.play()?;
    Ok((
        rx,
        level_rx,
        voiced_rx,
        debug_rx,
        spec_rx,
        sample_rate,
        stream,
        processing,
    ))
}

fn choose_input_config(
    device: &cpal::Device,
    target_rate: u32,
) -> anyhow::Result<cpal::SupportedStreamConfig> {
    let mut best: Option<cpal::SupportedStreamConfig> = None;
    let mut best_score = i32::MIN;
    for range in device.supported_input_configs()? {
        let min = range.min_sample_rate().0;
        let max = range.max_sample_rate().0;
        let rate = if target_rate >= min && target_rate <= max {
            target_rate
        } else {
            min
        };
        let cfg = range.with_sample_rate(cpal::SampleRate(rate));

        let mut score = 0;
        if cfg.sample_rate().0 == target_rate {
            score += 2;
        }
        if cfg.sample_format() == cpal::SampleFormat::F32 {
            score += 1;
        }

        if score > best_score {
            best_score = score;
            best = Some(cfg);
        }
    }
    best.ok_or_else(|| anyhow::anyhow!("no supported input config"))
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    mut producer: HeapProducer<f32>,
    err_fn: fn(cpal::StreamError),
) -> anyhow::Result<cpal::Stream>
where
    T: Sample + SizedSample,
    f32: FromSample<T>,
{
    let stream = device.build_input_stream(
        config,
        move |data: &[T], _| {
            for frame in data.chunks(channels) {
                let s = frame[0].to_sample::<f32>();
                let _ = producer.push(s);
            }
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

fn stream_error(err: cpal::StreamError) {
    eprintln!("stream error: {}", err);
}

#[derive(Clone)]
struct ProcessingConfig {
    frame_size: usize,
    hop_size: usize,
    proc_frame_size: usize,
    lpc_order: usize,
    input_sample_rate: u32,
    proc_sample_rate: u32,
    window: Arc<Vec<f32>>,
    rms_gate: f32,
    spec_nfft: usize,
    spec_bins: usize,
    spec_bin_hz: f64,
}

impl ProcessingConfig {
    const MAX_FORMANT_HZ: f64 = 5000.0;
    const DISPLAY_MAX_HZ: f64 = 2500.0;
    const PREEMPH_COEF: f32 = 0.97;
    const RMS_GATE: f32 = 0.05;
    const FRAME_SEC: f32 = 0.025;
    const HOP_SEC: f32 = 0.010;
    const SPEC_NFFT: usize = 2048;
    const SPECTRUM_DB_MIN: f32 = -80.0;
    const SPECTRUM_DB_MAX: f32 = 0.0;
    const LPC_ORDER: usize = 12;
    const FORMANT_BW_MAX: f64 = 1000.0;
    const FORMANT_FMIN: f64 = 90.0;
    const PROC_SAMPLE_RATE: u32 = 10_000;

    fn new(sample_rate: u32) -> Self {
        let frame_size = ((sample_rate as f32) * Self::FRAME_SEC) as usize; // input 25ms
        let hop_size = ((sample_rate as f32) * Self::HOP_SEC) as usize; // input 10ms
        let proc_sample_rate = Self::PROC_SAMPLE_RATE;
        let proc_frame_size = ((proc_sample_rate as f32) * Self::FRAME_SEC) as usize; // 25ms @ 10k
        let lpc_order = Self::LPC_ORDER;
        let window = Arc::new(hamming_window(proc_frame_size));
        let rms_gate = Self::RMS_GATE;
        let spec_nfft = Self::SPEC_NFFT;
        let max_bin = ((Self::DISPLAY_MAX_HZ / proc_sample_rate as f64) * spec_nfft as f64)
            .floor()
            .min((spec_nfft / 2) as f64) as usize;
        let spec_bins = max_bin + 1;
        let spec_bin_hz = (proc_sample_rate as f64) / (spec_nfft as f64);
        Self {
            frame_size,
            hop_size,
            proc_frame_size,
            lpc_order,
            input_sample_rate: sample_rate,
            proc_sample_rate,
            window,
            rms_gate,
            spec_nfft,
            spec_bins,
            spec_bin_hz,
        }
    }

    fn rms_gate_value() -> f32 {
        Self::RMS_GATE
    }
}

fn spawn_processing_thread(
    mut consumer: HeapConsumer<f32>,
    tx: Sender<FormantPoint>,
    level_tx: Sender<f32>,
    voiced_tx: Sender<bool>,
    debug_tx: Sender<DebugInfo>,
    spec_tx: Sender<SpectrumFrame>,
    start: Instant,
    cfg: ProcessingConfig,
) {
    thread::spawn(move || {
        let mut buffer = VecDeque::<f32>::with_capacity(cfg.frame_size * 2);
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(cfg.spec_nfft);
        let mut fft_input = vec![Complex32::new(0.0, 0.0); cfg.spec_nfft];
        let resample_fft = planner.plan_fft_forward(cfg.frame_size);
        let resample_ifft = planner.plan_fft_inverse(cfg.proc_frame_size);
        let mut resample_in = vec![Complex32::new(0.0, 0.0); cfg.frame_size];
        let mut resample_out = vec![Complex32::new(0.0, 0.0); cfg.proc_frame_size];
        loop {
            if consumer.len() < cfg.hop_size {
                thread::sleep(Duration::from_millis(1));
                continue;
            }

            for _ in 0..cfg.hop_size {
                if let Some(s) = consumer.pop() {
                    buffer.push_back(s);
                }
            }

            if buffer.len() >= cfg.frame_size {
                let frame: Vec<f32> = buffer.iter().take(cfg.frame_size).copied().collect();
                let rms = rms_level(&frame);
                let voiced = rms >= cfg.rms_gate;
                let _ = level_tx.try_send(rms);
                let _ = voiced_tx.try_send(voiced);

                let down = fft_resample(
                    &frame,
                    cfg.proc_frame_size,
                    resample_fft.as_ref(),
                    resample_ifft.as_ref(),
                    &mut resample_in,
                    &mut resample_out,
                );
                let x = preprocess_frame(&down, &cfg);
                let spectrum_db = spectrum_db(&x, &cfg, fft.as_ref(), &mut fft_input);
                let lpc = lpc_analysis(&x, &cfg);
                let t = start.elapsed().as_secs_f64();
                if voiced {
                    if let Some(lpc) = &lpc {
                        let mut in_range: Vec<f64> = lpc
                            .formants
                            .iter()
                            .copied()
                            .filter(|f| *f <= ProcessingConfig::DISPLAY_MAX_HZ)
                            .collect();
                        in_range.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        if in_range.len() >= 2 {
                            let _ = tx.try_send(FormantPoint {
                                t,
                                f1: in_range[0],
                                f2: in_range[1],
                            });
                        }
                        let _ = debug_tx.try_send(DebugInfo {
                            rms,
                            voiced,
                            peaks: lpc.peaks,
                            formants: lpc.formants.len(),
                        });
                    } else {
                        let _ = debug_tx.try_send(DebugInfo {
                            rms,
                            voiced,
                            peaks: 0,
                            formants: 0,
                        });
                    }
                } else {
                    let _ = debug_tx.try_send(DebugInfo {
                        rms,
                        voiced,
                        peaks: 0,
                        formants: 0,
                    });
                }

                let lpc_env_db = lpc
                    .map(|l| l.env_db)
                    .unwrap_or_else(|| vec![ProcessingConfig::SPECTRUM_DB_MIN; cfg.spec_bins]);
                let _ = spec_tx.try_send(SpectrumFrame {
                    t,
                    spectrum_db,
                    lpc_env_db,
                });

                for _ in 0..cfg.hop_size {
                    let _ = buffer.pop_front();
                }
            }
        }
    });
}

struct LpcAnalysis {
    formants: Vec<f64>,
    peaks: usize,
    env_db: Vec<f32>,
}

fn preprocess_frame(frame: &[f32], cfg: &ProcessingConfig) -> Vec<f32> {
    let mut x = frame.to_vec();

    // Pre-emphasis (x[n] - 0.97 * x[n-1])
    let mut prev = x[0];
    for i in 1..x.len() {
        let current = x[i];
        x[i] = x[i] - ProcessingConfig::PREEMPH_COEF * prev;
        prev = current;
    }

    // Window
    for (v, w) in x.iter_mut().zip(cfg.window.iter()) {
        *v *= *w;
    }

    x
}

fn fft_resample(
    x: &[f32],
    out_len: usize,
    fft: &dyn Fft<f32>,
    ifft: &dyn Fft<f32>,
    in_buf: &mut [Complex32],
    out_buf: &mut [Complex32],
) -> Vec<f32> {
    let in_len = x.len();
    if in_len == 0 || out_len == 0 {
        return Vec::new();
    }

    for (i, v) in in_buf.iter_mut().enumerate() {
        let sample = if i < in_len { x[i] } else { 0.0 };
        *v = Complex32::new(sample, 0.0);
    }
    fft.process(in_buf);

    for v in out_buf.iter_mut() {
        *v = Complex32::new(0.0, 0.0);
    }

    let in_half = in_len / 2;
    let out_half = out_len / 2;
    let k_max = in_half.min(out_half);

    out_buf[0] = in_buf[0];
    for k in 1..=k_max {
        out_buf[k] = in_buf[k];
        out_buf[out_len - k] = in_buf[in_len - k];
    }
    if in_len % 2 == 0 && out_len % 2 == 0 {
        out_buf[out_half] = in_buf[in_half];
    }

    ifft.process(out_buf);

    let scale = 1.0 / (in_len as f32);
    out_buf.iter().map(|c| c.re * scale).collect()
}

fn spectrum_db(
    x: &[f32],
    cfg: &ProcessingConfig,
    fft: &dyn Fft<f32>,
    fft_input: &mut [Complex32],
) -> Vec<f32> {
    for (i, v) in fft_input.iter_mut().enumerate() {
        let sample = if i < x.len() { x[i] } else { 0.0 };
        *v = Complex32::new(sample, 0.0);
    }
    fft.process(fft_input);

    let mut mags = Vec::with_capacity(cfg.spec_bins);
    for k in 0..cfg.spec_bins {
        let mag = fft_input[k].norm().max(1e-9);
        mags.push(20.0 * mag.log10());
    }
    mags
}

fn lpc_analysis(x: &[f32], cfg: &ProcessingConfig) -> Option<LpcAnalysis> {
    let r_full = autocorrelation_full(x);
    let max_lag = x.len().saturating_sub(1);
    if r_full.len() < max_lag + 1 + cfg.lpc_order {
        return None;
    }
    let r = &r_full[max_lag..max_lag + 1 + cfg.lpc_order];
    let a = levinson_durbin(r, cfg.lpc_order)?;
    let env_linear = lpc_envelope(&a, cfg.proc_sample_rate, cfg.spec_nfft);
    let formants = lpc_formants_from_coeffs(
        &a,
        cfg.proc_sample_rate,
        ProcessingConfig::FORMANT_FMIN,
        ProcessingConfig::MAX_FORMANT_HZ,
        ProcessingConfig::FORMANT_BW_MAX,
    );
    let peaks = formants.len();
    let mut env_db: Vec<f32> = env_linear.into_iter().map(lin_to_db).collect();
    env_db.truncate(cfg.spec_bins);
    Some(LpcAnalysis {
        formants,
        peaks,
        env_db,
    })
}

fn rms_level(x: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    for &v in x {
        sum += (v as f64) * (v as f64);
    }
    let rms = (sum / x.len() as f64).sqrt();
    rms as f32
}

fn autocorrelation_full(x: &[f32]) -> Vec<f64> {
    let mean = x.iter().copied().sum::<f32>() / x.len() as f32;
    let n = x.len();
    let mut r = vec![0.0f64; 2 * n - 1];
    // lags from -(n-1) .. (n-1), centered at index n-1
    for lag in -(n as isize - 1)..=(n as isize - 1) {
        let mut acc = 0.0f64;
        for i in 0..n {
            let j = i as isize + lag;
            if j >= 0 && (j as usize) < n {
                let a = (x[i] - mean) as f64;
                let b = (x[j as usize] - mean) as f64;
                acc += a * b;
            }
        }
        r[(lag + (n as isize - 1)) as usize] = acc;
    }
    r
}

fn levinson_durbin(r: &[f64], order: usize) -> Option<Vec<f64>> {
    if r.len() < order + 1 || r[0] == 0.0 {
        return None;
    }

    let mut a = vec![0.0f64; order + 1];
    a[0] = 1.0;
    let mut e = r[0];

    for i in 1..=order {
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = -acc / e;
        let a_prev = a.clone();
        a[i] = k;
        for j in 1..i {
            a[j] = a_prev[j] + k * a_prev[i - j];
        }
        e *= 1.0 - k * k;
        if e <= 0.0 {
            return None;
        }
    }

    Some(a)
}

fn lpc_envelope(a: &[f64], _sample_rate: u32, nfft: usize) -> Vec<f32> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(nfft);
    let mut input: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); nfft];
    for (i, v) in a.iter().enumerate() {
        if i < nfft {
            input[i] = Complex64::new(*v, 0.0);
        }
    }
    fft.process(&mut input);

    let half = nfft / 2;
    let mut env = Vec::with_capacity(half + 1);
    for k in 0..=half {
        let mag = input[k].norm().max(1e-9);
        env.push((1.0 / mag) as f32);
    }
    env
}

fn lpc_formants_from_coeffs(
    a: &[f64],
    sample_rate: u32,
    fmin: f64,
    fmax: f64,
    bw_max: f64,
) -> Vec<f64> {
    if a.len() < 2 || a[0].abs() < 1e-12 {
        return Vec::new();
    }

    let roots = durand_kerner_roots(a, 60, 1e-8);
    let mut formants = Vec::new();
    for z in roots.iter() {
        let r = z.norm();
        if r >= 1.0 || z.im <= 0.0 {
            continue;
        }
        let angle = z.arg();
        let freq = angle * (sample_rate as f64) / (2.0 * PI);
        let bw = -(sample_rate as f64) / PI * r.ln();
        if freq > fmin && freq < fmax && bw < bw_max {
            formants.push(freq);
        }
    }
    formants.sort_by(|a, b| a.partial_cmp(b).unwrap());
    formants
}

fn durand_kerner_roots(a: &[f64], max_iter: usize, tol: f64) -> Vec<Complex64> {
    let n = a.len().saturating_sub(1);
    if n == 0 {
        return Vec::new();
    }

    let radius = 0.9;
    let two_pi = 2.0 * PI;
    let mut roots: Vec<Complex64> = (0..n)
        .map(|k| {
            let theta = two_pi * (k as f64) / (n as f64);
            Complex64::new(radius * theta.cos(), radius * theta.sin())
        })
        .collect();

    for _ in 0..max_iter {
        let mut converged = true;
        for i in 0..n {
            let mut denom = Complex64::new(1.0, 0.0);
            for j in 0..n {
                if i != j {
                    denom *= roots[i] - roots[j];
                }
            }
            let p = poly_eval(a, roots[i]);
            let delta = if denom.norm() < 1e-12 {
                Complex64::new(1e-6, 1e-6)
            } else {
                p / denom
            };
            let next = roots[i] - delta;
            if (next - roots[i]).norm() > tol {
                converged = false;
            }
            roots[i] = next;
        }
        if converged {
            break;
        }
    }

    roots
}

fn poly_eval(a: &[f64], z: Complex64) -> Complex64 {
    let mut acc = Complex64::new(a[0], 0.0);
    for &coef in &a[1..] {
        acc = acc * z + Complex64::new(coef, 0.0);
    }
    acc
}

fn lin_to_db(value: f32) -> f32 {
    20.0 * value.max(1e-9).log10()
}


fn hamming_window(n: usize) -> Vec<f32> {
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let val = 0.54 - 0.46 * ((2.0 * PI * i as f64) / (n as f64 - 1.0)).cos();
        w.push(val as f32);
    }
    w
}

fn main() -> anyhow::Result<()> {
    let options = eframe::NativeOptions::default();
    let app = FormantApp::new()?;
    if let Err(err) = eframe::run_native("Formant GUI", options, Box::new(|_| Box::new(app))) {
        return Err(anyhow::anyhow!(err.to_string()));
    }
    Ok(())
}
