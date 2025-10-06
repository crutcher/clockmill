use burn::Tensor;
use burn::prelude::{Backend, Bool};
use clockmill::simulations::surface::conway::Conway;
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
        conway: Conway<B>,
        noise: f64,
        step_duration: Duration,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new(conway.state.clone()));

        let shutdown_clone = shutdown.clone();
        let state_clone = state.clone();

        let handle = thread::spawn(move || {
            let mut conway = conway;

            while !shutdown_clone.load(Ordering::Relaxed) {
                // Update simulation
                conway.fuzz(noise);
                conway.wrap();
                conway.step_no_wrap();

                // Export
                *state_clone.lock().unwrap() = conway.state.clone();

                thread::sleep(step_duration);
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
