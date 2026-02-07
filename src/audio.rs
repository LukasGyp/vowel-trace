use std::sync::Arc;
use std::time::Instant;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};
use crossbeam_channel::Receiver;
use ringbuf::{HeapProducer, HeapRb};

use crate::processing::{ProcessingConfig, spawn_processing_thread};
use crate::tracking::TrackingParams;
use crate::types::{DebugInfo, FormantPoint, SpectrumFrame};

pub(crate) fn setup_audio() -> anyhow::Result<(
    Receiver<FormantPoint>,
    Receiver<f32>,
    Receiver<bool>,
    Receiver<DebugInfo>,
    Receiver<SpectrumFrame>,
    Arc<TrackingParams>,
    String,
    u32,
    cpal::Stream,
    ProcessingConfig,
)> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no input device available"))?;
    let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
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
    let tracking_params = Arc::new(TrackingParams::new());

    let stream_config: cpal::StreamConfig = config.clone().into();
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => build_stream::<f32>,
        cpal::SampleFormat::I16 => build_stream::<i16>,
        cpal::SampleFormat::U16 => build_stream::<u16>,
        _ => return Err(anyhow::anyhow!("unsupported sample format")),
    }(&device, &stream_config, channels, producer, stream_error)?;

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
        tracking_params.clone(),
    );

    stream.play()?;
    Ok((
        rx,
        level_rx,
        voiced_rx,
        debug_rx,
        spec_rx,
        tracking_params,
        device_name,
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
