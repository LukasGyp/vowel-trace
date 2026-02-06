mod app;
mod audio;
mod processing;
mod types;

use app::FormantApp;

fn main() -> anyhow::Result<()> {
    let options = eframe::NativeOptions::default();
    let app = FormantApp::new()?;
    if let Err(err) = eframe::run_native("Formant GUI", options, Box::new(|_| Box::new(app))) {
        return Err(anyhow::anyhow!(err.to_string()));
    }
    Ok(())
}
