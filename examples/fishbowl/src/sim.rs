use burn::Tensor;
use burn::prelude::{Backend, Bool};
use clockmill::simulations::surface::conway::life2d::ConwayLife2DState;
use indicatif::ProgressBar;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub struct Simulation<B: Backend> {
    handle: Option<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    pub state: Arc<Mutex<Tensor<B, 2, Bool>>>,
}

impl<B: Backend> Simulation<B> {
    pub fn new(
        conway: ConwayLife2DState<B>,
        noise: f64,
        step_duration: Option<Duration>,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new(conway.state.clone()));

        let shutdown_clone = shutdown.clone();
        let state_clone = state.clone();

        let handle = thread::spawn(move || {
            let mut conway = conway;

            let progress = ProgressBar::new_spinner();
            let delay_smoothing = 20;
            let mut avg_delay = std::time::Duration::from_secs_f32(0.0);
            let mut last_time = std::time::Instant::now();

            while !shutdown_clone.load(Ordering::Relaxed) {
                {
                    let now = std::time::Instant::now();
                    let dt = now - last_time;
                    avg_delay = (avg_delay * delay_smoothing + dt) / (delay_smoothing + 1);
                    last_time = now;
                }
                let avg_tps = 1.0 / avg_delay.as_secs_f32();
                progress.set_message(format!("sim:{:.0}tps", avg_tps));
                progress.tick();

                let t0 = std::time::Instant::now();

                // Update simulation
                conway.fuzz(noise);
                conway.step();

                // Export
                *state_clone.lock().unwrap() = conway.state.clone();

                let t1 = std::time::Instant::now();

                let update_delay = t1.duration_since(t0);

                if let Some(step_duration) = step_duration
                    && step_duration > update_delay
                {
                    let sleep_duration = step_duration - update_delay;
                    thread::sleep(sleep_duration);
                }
            }
        });

        Simulation {
            handle: Some(handle),
            shutdown,
            state,
        }
    }

    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.join().unwrap();
        }
    }
}
