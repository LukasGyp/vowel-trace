mod app;
mod audio;
mod processing;
mod tracking;
mod types;

use app::FormantApp;
use eframe::egui;

fn main() -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 600.0]),
        ..Default::default()
    };
    let app = FormantApp::new()?;
    if let Err(err) = eframe::run_native("Formant GUI", options, Box::new(|_| Box::new(app))) {
        return Err(anyhow::anyhow!(err.to_string()));
    }
    Ok(())
}
