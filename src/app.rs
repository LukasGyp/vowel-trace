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
    input_device_name: String,
}

impl FormantApp {
    pub(crate) fn new() -> anyhow::Result<Self> {
        let (
            rx,
            level_rx,
            voiced_rx,
            debug_rx,
            spec_rx,
            tracking_params,
            input_device_name,
            sample_rate,
            stream,
            cfg,
        ) =
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
            input_device_name,
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

        let rms_max = ProcessingConfig::RMS_GATE * 2.0;
        let rms_norm = if rms_max > 1e-6 {
            (self.last_rms / rms_max).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let gate_norm = if rms_max > 1e-6 {
            (self.last_rms_gate / rms_max).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let voiced_color = if self.last_voiced {
            egui::Color32::from_rgb(80, 180, 90)
        } else {
            egui::Color32::from_rgb(200, 80, 80)
        };
        let voiced_text = if self.last_voiced {
            "Voiced"
        } else {
            "Unvoiced"
        };

        egui::Area::new("rms_status".into())
            .anchor(egui::Align2::RIGHT_TOP, [-12.0, 12.0])
            .show(ctx, |ui| {
                ui.group(|ui| {
                    ui.set_min_width(220.0);
                    ui.set_max_width(220.0);
                    ui.label(
                        egui::RichText::new(voiced_text)
                            .color(voiced_color)
                            .strong(),
                    );
                    let bar_size = egui::vec2(200.0, 18.0);
                    let (rect, _resp) = ui.allocate_exact_size(bar_size, egui::Sense::hover());
                    let painter = ui.painter();
                    let rounding = egui::Rounding::same(3.0);
                    painter.rect_filled(rect, rounding, egui::Color32::from_gray(30));
                    let filled = egui::Rect::from_min_max(
                        rect.min,
                        egui::pos2(
                            rect.min.x + rect.width() * rms_norm,
                            rect.max.y,
                        ),
                    );
                    painter.rect_filled(filled, rounding, egui::Color32::from_rgb(80, 160, 220));
                    let gate_x = rect.min.x + rect.width() * gate_norm;
                    painter.line_segment(
                        [egui::pos2(gate_x, rect.min.y), egui::pos2(gate_x, rect.max.y)],
                        egui::Stroke::new(1.5, egui::Color32::from_rgb(230, 200, 60)),
                    );
                    painter.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        format!("RMS {:.3}", self.last_rms),
                        egui::TextStyle::Small.resolve(ui.style()),
                        egui::Color32::WHITE,
                    );
                    ui.add_space(4.0);
                    ui.label(format!("threshold {:.4}", self.last_rms_gate));
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Real-time Formant Tracker");
            ui.label(format!(
                "sample_rate: {} Hz (proc {} Hz)",
                self.sample_rate,
                ProcessingConfig::PROC_SAMPLE_RATE
            ));
            ui.label(format!("input device: {}", self.input_device_name));
            ui.label(format!(
                "params: win 25ms, hop 10ms, max_formant {}Hz, preemph {}, lpc_order {}",
                ProcessingConfig::MAX_FORMANT_HZ,
                ProcessingConfig::PREEMPH_COEF,
                ProcessingConfig::LPC_ORDER
            ));
            // last detect removed

            ui.separator();
            ui.label("tracking controls");
            let mut mode = self.tracking_params.mode();
            egui::ComboBox::from_label("mode")
                .selected_text(mode.label())
                .show_ui(ui, |ui| {
                    let labels = [
                        TrackingMode::Raw.label(),
                        TrackingMode::Kalman.label(),
                        TrackingMode::Viterbi.label(),
                    ];
                    let font_id = egui::TextStyle::Button.resolve(ui.style());
                    let max_width = ui.fonts(|fonts| {
                        labels
                            .iter()
                            .map(|label| {
                                fonts
                                    .layout_no_wrap(label.to_string(), font_id.clone(), ui.visuals().text_color())
                                    .size()
                                    .x
                            })
                            .fold(0.0f32, f32::max)
                    });
                    let target_width = max_width + ui.spacing().button_padding.x * 2.0;

                    let clicked = ui
                        .add_sized(
                            [target_width, 0.0],
                            egui::SelectableLabel::new(mode == TrackingMode::Raw, labels[0]),
                        )
                        .clicked();
                    if clicked {
                        mode = TrackingMode::Raw;
                        self.tracking_params.set_mode(mode);
                    }
                    let clicked = ui
                        .add_sized(
                            [target_width, 0.0],
                            egui::SelectableLabel::new(mode == TrackingMode::Kalman, labels[1]),
                        )
                        .clicked();
                    if clicked {
                        mode = TrackingMode::Kalman;
                        self.tracking_params.set_mode(mode);
                    }
                    let clicked = ui
                        .add_sized(
                            [target_width, 0.0],
                            egui::SelectableLabel::new(mode == TrackingMode::Viterbi, labels[2]),
                        )
                        .clicked();
                    if clicked {
                        mode = TrackingMode::Viterbi;
                        self.tracking_params.set_mode(mode);
                    }
                });

            match mode {
                TrackingMode::Kalman => {
                    let mut kalman_q = self.tracking_params.kalman_q();
                    if ui
                        .add(
                            egui::Slider::new(&mut kalman_q, 1000.0..=2_000_000.0)
                                .text("kalman Q"),
                        )
                        .changed()
                    {
                        self.tracking_params.set_kalman_q(kalman_q);
                    }
                    let mut kalman_r = self.tracking_params.kalman_r();
                    if ui
                        .add(
                            egui::Slider::new(&mut kalman_r, 1.0..=5000.0).text("kalman R"),
                        )
                        .changed()
                    {
                        self.tracking_params.set_kalman_r(kalman_r);
                    }
                    let mut max_jump = self.tracking_params.kalman_max_jump_hz();
                    if ui
                        .add(
                            egui::Slider::new(&mut max_jump, 100.0..=1500.0)
                                .text("kalman max jump (Hz)"),
                        )
                        .changed()
                    {
                        self.tracking_params.set_kalman_max_jump_hz(max_jump);
                    }
                }
                TrackingMode::Viterbi => {
                    let mut v_wt = self.tracking_params.viterbi_transition_smoothness();
                    if ui
                        .add(
                            egui::Slider::new(&mut v_wt, 0.0..=0.02).text("viterbi transition"),
                        )
                        .changed()
                    {
                        self.tracking_params
                            .set_viterbi_transition_smoothness(v_wt);
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
                        .add(
                            egui::Slider::new(&mut v_ws, 0.0..=5.0).text("viterbi strength"),
                        )
                        .changed()
                    {
                        self.tracking_params.set_viterbi_strength_weight(v_ws);
                    }
                }
                TrackingMode::Raw => {}
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
                            let thick_targets = [1000.0, 2000.0, 3000.0];
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
                            let mut found = [false, false];
                            for mark in &mut marks {
                                let mut is_thick = false;
                                let targets = [200.0, 1200.0];
                                for (idx, target) in targets.iter().enumerate() {
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
                            let targets = [200.0, 1200.0];
                            for (idx, value) in targets.iter().enumerate() {
                                if !found[idx] {
                                    marks.push(GridMark {
                                        value: *value,
                                        step_size: thick_step,
                                    });
                                }
                            }
                            marks
                        }
                    })
                    .label_formatter(|_name, _value| String::new())
                    .coordinates_formatter(
                        egui_plot::Corner::LeftBottom,
                        CoordinatesFormatter::new(|value, _bounds| {
                            let x = 4000.0 - value.x;
                            let y = 1200.0 - value.y;
                            format!("F2: {:.0}\nF1: {:.0}", x, y)
                        }),
                    )
                    .include_x(500.0)
                    .include_x(3500.0)
                    .include_y(0.0)
                    .include_y(1200.0)
                    .x_axis_formatter(|mark: GridMark, _max_char, _range| {
                        format!("{:.0}", 4000.0 - mark.value)
                    })
                    .y_axis_formatter(|mark: GridMark, _max_char, _range| {
                        format!("{:.0}", 1200.0 - mark.value)
                    });

                f1f2_plot.show(left, |plot_ui| {
                    plot_ui.set_auto_bounds(egui::Vec2b::new(false, false));
                    plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
                        [500.0, 0.0],
                        [3500.0, 1200.0],
                    ));
                    let gap = 0.3;
                    if let Some(p) = self.points.last() {
                        if now - p.t <= gap {
                            let pts = vec![[4000.0 - p.f2, 1200.0 - p.f1]];
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
