use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use eframe::egui;
use egui_plot::{CoordinatesFormatter, GridMark, Line, Plot, PlotPoints, Points};

use crate::audio::setup_audio;
use crate::processing::ProcessingConfig;
use crate::tracking::TrackingParams;
use crate::tracking::tracker::TrackingMode;
use crate::types::{DebugInfo, FormantPoint, SpectrumFrame};

pub(crate) struct FormantApp {
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
    last_rms_gate: f32,
    last_detect: Option<f64>,
    last_voiced: bool,
    last_debug: Option<DebugInfo>,
    tracking_params: std::sync::Arc<TrackingParams>,
}

impl FormantApp {
    pub(crate) fn new() -> anyhow::Result<Self> {
        let (rx, level_rx, voiced_rx, debug_rx, spec_rx, tracking_params, sample_rate, stream, cfg) =
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
            last_rms_gate: ProcessingConfig::RMS_GATE,
            last_detect: None,
            last_voiced: false,
            last_debug: None,
            tracking_params,
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
            self.last_rms_gate = d.rms_gate;
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
                self.last_rms_gate
            ));
            if let Some(d) = self.last_debug {
                ui.label(format!("debug: peaks {}, formants {}", d.peaks, d.formants));
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

            ui.separator();
            ui.label("tracking controls");
            let mut mode = self.tracking_params.mode();
            egui::ComboBox::from_label("mode")
                .selected_text(mode.label())
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_value(&mut mode, TrackingMode::Raw, "Raw")
                        .changed()
                    {
                        self.tracking_params.set_mode(mode);
                    }
                    if ui
                        .selectable_value(&mut mode, TrackingMode::Kalman, "Kalman")
                        .changed()
                    {
                        self.tracking_params.set_mode(mode);
                    }
                    if ui
                        .selectable_value(&mut mode, TrackingMode::Viterbi, "Viterbi")
                        .changed()
                    {
                        self.tracking_params.set_mode(mode);
                    }
                });

            let mut kalman_q = self.tracking_params.kalman_q();
            if ui
                .add(egui::Slider::new(&mut kalman_q, 1000.0..=200000.0).text("kalman Q"))
                .changed()
            {
                self.tracking_params.set_kalman_q(kalman_q);
            }
            let mut kalman_r = self.tracking_params.kalman_r();
            if ui
                .add(egui::Slider::new(&mut kalman_r, 10.0..=5000.0).text("kalman R"))
                .changed()
            {
                self.tracking_params.set_kalman_r(kalman_r);
            }
            let mut max_jump = self.tracking_params.kalman_max_jump_hz();
            if ui
                .add(egui::Slider::new(&mut max_jump, 100.0..=1500.0).text("kalman max jump (Hz)"))
                .changed()
            {
                self.tracking_params.set_kalman_max_jump_hz(max_jump);
            }

            let mut v_wt = self.tracking_params.viterbi_transition_smoothness();
            if ui
                .add(egui::Slider::new(&mut v_wt, 0.0..=0.02).text("viterbi transition"))
                .changed()
            {
                self.tracking_params.set_viterbi_transition_smoothness(v_wt);
            }
            let mut v_wd = self.tracking_params.viterbi_dropout_penalty();
            if ui
                .add(egui::Slider::new(&mut v_wd, 0.0..=10.0).text("viterbi dropout"))
                .changed()
            {
                self.tracking_params.set_viterbi_dropout_penalty(v_wd);
            }
            let mut v_ws = self.tracking_params.viterbi_strength_weight();
            if ui
                .add(egui::Slider::new(&mut v_ws, 0.0..=5.0).text("viterbi strength"))
                .changed()
            {
                self.tracking_params.set_viterbi_strength_weight(v_ws);
            }

            let f1 = self.formant_line("F1", |p| p.f1);
            let f2 = self.formant_line("F2", |p| p.f2);

            ui.columns(3, |cols| {
                let (left, rest) = cols.split_at_mut(1);
                let (middle, right) = rest.split_at_mut(1);
                let left = &mut left[0];
                let middle = &mut middle[0];
                let right = &mut right[0];

                let f1f2_plot = Plot::new("f1f2")
                    .legend(egui_plot::Legend::default())
                    .height(400.0)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .x_axis_label("F2")
                    .y_axis_label("F1")
                    .x_grid_spacer({
                        let default_spacer = egui_plot::log_grid_spacer(10);
                        move |input| {
                            let mut marks = default_spacer(input);
                            if marks.is_empty() {
                                return marks;
                            }
                            let mut min_step = f64::INFINITY;
                            let mut max_step: f64 = 0.0;
                            for mark in &marks {
                                min_step = min_step.min(mark.step_size);
                                max_step = max_step.max(mark.step_size);
                            }
                            let thick_step = max_step.max(min_step * 10.0);
                            let thick_targets = [600.0, 1600.0, 2600.0];
                            let mut found = [false, false, false];
                            for mark in &mut marks {
                                let mut is_thick = false;
                                for (idx, target) in thick_targets.iter().enumerate() {
                                    if (mark.value - target).abs() < 1e-6 {
                                        found[idx] = true;
                                        is_thick = true;
                                        break;
                                    }
                                }
                                if is_thick {
                                    mark.step_size = thick_step;
                                } else {
                                    mark.step_size = min_step;
                                }
                            }
                            for (idx, target) in thick_targets.iter().enumerate() {
                                if !found[idx] {
                                    marks.push(GridMark {
                                        value: *target,
                                        step_size: thick_step,
                                    });
                                }
                            }
                            marks
                        }
                    })
                    .y_grid_spacer({
                        let default_spacer = egui_plot::log_grid_spacer(10);
                        move |input| {
                            let mut marks = default_spacer(input);
                            if marks.is_empty() {
                                return marks;
                            }
                            let mut min_step = f64::INFINITY;
                            let mut max_step: f64 = 0.0;
                            for mark in &marks {
                                min_step = min_step.min(mark.step_size);
                                max_step = max_step.max(mark.step_size);
                            }
                            let thick_step = max_step.max(min_step * 10.0);
                            let mut has_zero = false;
                            for mark in &mut marks {
                                if (mark.value - 1200.0).abs() < 1e-6 {
                                    mark.step_size = thick_step;
                                    has_zero = true;
                                } else {
                                    mark.step_size = min_step;
                                }
                            }
                            if !has_zero {
                                marks.push(GridMark {
                                    value: 1200.0,
                                    step_size: thick_step,
                                });
                            }
                            marks
                        }
                    })
                    .label_formatter(|_name, _value| String::new())
                    .coordinates_formatter(
                        egui_plot::Corner::LeftBottom,
                        CoordinatesFormatter::new(|value, _bounds| {
                            let x = 3600.0 - value.x;
                            let y = 1200.0 - value.y;
                            format!("F2: {:.0}\nF1: {:.0}", x, y)
                        }),
                    )
                    .include_x(600.0)
                    .include_x(3100.0)
                    .include_y(200.0)
                    .include_y(1200.0)
                    .x_axis_formatter(|mark: GridMark, _max_char, _range| {
                        format!("{:.0}", 3600.0 - mark.value)
                    })
                    .y_axis_formatter(|mark: GridMark, _max_char, _range| {
                        format!("{:.0}", 1200.0 - mark.value)
                    });

                f1f2_plot.show(left, |plot_ui| {
                    plot_ui.set_auto_bounds(egui::Vec2b::new(false, false));
                    plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
                        [600.0, 200.0],
                        [3100.0, 1200.0],
                    ));
                    let gap = 0.3;
                    if let Some(p) = self.points.last() {
                        if now - p.t <= gap {
                            let pts = vec![[3600.0 - p.f2, 1200.0 - p.f1]];
                            plot_ui.points(Points::new(pts).name("F1/F2").radius(6.0));
                        }
                    }
                });

                Plot::new("formants")
                    .legend(egui_plot::Legend::default())
                    .height(400.0)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .label_formatter(|_name, _value| String::new())
                    .coordinates_formatter(
                        egui_plot::Corner::LeftBottom,
                        CoordinatesFormatter::new(|value, _bounds| {
                            format!("t: {:.2}\nHz: {:.0}", value.x, value.y)
                        }),
                    )
                    .include_y(0.0)
                    .include_y(ProcessingConfig::DISPLAY_MAX_HZ)
                    .include_x(now - window_sec)
                    .include_x(now)
                    .show(middle, |plot_ui| {
                        plot_ui.line(f1);
                        plot_ui.line(f2);
                    });

                let spectrogram_plot = Plot::new("spectrogram")
                    .legend(egui_plot::Legend::default())
                    .height(400.0)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .label_formatter(|_name, _value| String::new())
                    .coordinates_formatter(
                        egui_plot::Corner::LeftBottom,
                        CoordinatesFormatter::new(|value, _bounds| {
                            format!("Hz: {:.0}\ndB: {:.1}", value.x, value.y)
                        }),
                    )
                    .include_x(0.0)
                    .include_x(ProcessingConfig::DISPLAY_MAX_HZ)
                    .include_y(-100.0)
                    .include_y(30.0);

                spectrogram_plot.show(right, |plot_ui| {
                    plot_ui.set_auto_bounds(egui::Vec2b::new(false, false));
                    plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
                        [0.0, -100.0],
                        [ProcessingConfig::DISPLAY_MAX_HZ, 30.0],
                    ));
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
