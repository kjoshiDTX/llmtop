//! State shared between the polling tasks and the renderer.
//!
//! Each backend runs in its own task and writes its latest result here; the
//! renderer reads it. `Arc` gives every task a handle to the same value, and
//! `RwLock` lets many readers or one writer in at a time. It's the std lock,
//! not tokio's, because no one holds it across an `.await`: lock, copy or
//! write, unlock.

use crate::backend::Backend;
use crate::snapshot::Snapshot;
use crate::system::{SystemSampler, SystemStats};
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::watch;
use tokio::task::JoinHandle;

#[derive(Debug, Clone)]
pub enum BackendStatus {
    /// Registered, but the first poll hasn't finished.
    Pending,
    Up(Snapshot),
    Down(String),
}

#[derive(Debug, Default)]
pub struct State {
    pub system: Option<SystemStats>,
    /// Keyed by backend name, so the table order is stable.
    pub backends: BTreeMap<String, BackendStatus>,
}

pub struct Shared {
    pub state: RwLock<State>,
    /// Bumped after every write. Receivers wake on `changed()` and mark
    /// everything up to now as seen with `borrow_and_update()`, so a burst
    /// of writes produces one redraw.
    pub changed: watch::Sender<()>,
}

impl Default for Shared {
    fn default() -> Self {
        Self { state: RwLock::default(), changed: watch::Sender::new(()) }
    }
}

impl Shared {
    /// Runs `f` with write access, then wakes the renderer.
    ///
    /// `.expect` on the lock: it only fails if a thread panicked while
    /// holding it, and then the state may be half-written, so stopping is right.
    fn update(&self, f: impl FnOnce(&mut State)) {
        f(&mut self.state.write().expect("state lock poisoned"));
        // `send_replace` succeeds even with no receivers, e.g. in one-shot mode.
        self.changed.send_replace(());
    }
}

/// Starts a task that polls `backend` once, or every `every` if given.
///
/// Generic over `B` rather than taking `Box<dyn Backend>`: a trait whose
/// method returns `impl Future` can't be used as a trait object, and here we
/// don't need one, since each call is made with a known concrete type.
pub fn spawn_poller<B: Backend + 'static>(
    mut backend: B,
    shared: Arc<Shared>,
    every: Option<Duration>,
) -> JoinHandle<()> {
    let name = backend.name().to_string();
    shared.update(|s| {
        s.backends.insert(name.clone(), BackendStatus::Pending);
    });

    // `async move` moves `backend`, `shared` and `name` into the task, which
    // must own everything it uses because it may outlive this function.
    tokio::spawn(async move {
        let mut ticker = every.map(tokio::time::interval);
        loop {
            if let Some(t) = ticker.as_mut() {
                t.tick().await;
            }
            let status = match backend.poll().await {
                Ok(snapshot) => BackendStatus::Up(snapshot),
                Err(err) => BackendStatus::Down(format!("{err:#}")),
            };
            shared.update(|s| {
                s.backends.insert(name.clone(), status);
            });
            if ticker.is_none() {
                break;
            }
        }
    })
}

/// Same shape as `spawn_poller`, for host CPU and RAM.
pub fn spawn_system_sampler(shared: Arc<Shared>, every: Option<Duration>) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut sampler = SystemSampler::new();
        let mut ticker = every.map(tokio::time::interval);
        loop {
            if let Some(t) = ticker.as_mut() {
                t.tick().await;
            }
            let stats = sampler.sample().await;
            shared.update(|s| s.system = Some(stats));
            if ticker.is_none() {
                break;
            }
        }
    })
}
