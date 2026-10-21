//! State shared between the polling tasks and the renderer.
//!
//! Each backend runs in its own task and writes its latest result here; the
//! renderer reads it. `Arc` gives every task a handle to the same value, and
//! `RwLock` lets many readers or one writer in at a time. It's the std lock,
//! not tokio's, because no one holds it across an `.await`: lock, copy or
//! write, unlock.

use crate::backend::{Backend, BackendId};
use crate::detect::{Detector, Event, Thresholds};
use crate::record::Recorder;
use crate::snapshot::Snapshot;
use crate::system::{SystemSampler, SystemStats};
use chrono::{DateTime, Local};
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tokio::task::JoinHandle;

/// How many alerts to keep; older ones are dropped.
const MAX_ALERTS: usize = 100;

#[derive(Debug, Clone)]
pub struct Alert {
    pub at: DateTime<Local>,
    pub backend: BackendId,
    pub event: Event,
}

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
    /// A BTreeMap keeps the table order stable between frames.
    pub backends: BTreeMap<BackendId, BackendStatus>,
    /// The most recent alerts, oldest first.
    pub alerts: VecDeque<Alert>,
    /// Every alert since startup, counted per backend and kind of event, for
    /// the exporter. Unlike `alerts`, nothing is dropped.
    pub alert_counts: BTreeMap<(BackendId, &'static str), u64>,
}

impl State {
    fn push_alert(&mut self, alert: Alert) {
        *self.alert_counts.entry((alert.backend.clone(), alert.event.name())).or_default() += 1;
        if self.alerts.len() == MAX_ALERTS {
            self.alerts.pop_front();
        }
        self.alerts.push_back(alert);
    }
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

/// What every polling task needs besides its backend.
#[derive(Clone)]
pub struct PollConfig {
    /// Poll once if `None`.
    pub every: Option<Duration>,
    pub thresholds: Thresholds,
    pub recorder: Option<Arc<Recorder>>,
}

/// Starts a task that polls `backend` once, or every `config.every`.
///
/// Generic over `B` rather than taking `Box<dyn Backend>`: a trait whose
/// method returns `impl Future` can't be used as a trait object, and here we
/// don't need one, since each call is made with a known concrete type.
pub fn spawn_poller<B: Backend + 'static>(
    mut backend: B,
    shared: Arc<Shared>,
    config: PollConfig,
) -> JoinHandle<()> {
    let id = backend.id().clone();
    shared.update(|s| {
        s.backends.insert(id.clone(), BackendStatus::Pending);
        for name in Event::NAMES {
            s.alert_counts.insert((id.clone(), name), 0);
        }
    });

    // `async move` moves `backend`, `shared`, `id` and `config` into the
    // task, which must own everything it uses because it may outlive this
    // function. The detector is created inside, so the task alone owns it.
    tokio::spawn(async move {
        let mut detector = Detector::new(config.thresholds);
        let mut ticker = config.every.map(tokio::time::interval);
        loop {
            if let Some(t) = ticker.as_mut() {
                t.tick().await;
            }
            let fetched = backend.fetch().await;
            let now = Instant::now();
            if let Some(recorder) = &config.recorder
                && let Err(err) = recorder.write(&id, now, &fetched)
            {
                eprintln!("llmtop: recording failed: {err:#}");
            }

            let result = fetched.and_then(|raw| backend.parse(&raw, now));
            let events = detector.observe(&result, now);
            let status = match result {
                Ok(snapshot) => BackendStatus::Up(snapshot),
                Err(err) => BackendStatus::Down(format!("{err:#}")),
            };
            shared.update(|s| {
                s.backends.insert(id.clone(), status);
                let at = Local::now();
                for event in events {
                    s.push_alert(Alert { at, backend: id.clone(), event });
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alert_counts_survive_the_alert_list_cap() {
        let id = BackendId { kind: "vllm", instance: "x".into() };
        let mut state = State::default();
        for _ in 0..MAX_ALERTS + 5 {
            let event = Event::KvCacheHigh { model: "m".into(), usage: 0.95 };
            state.push_alert(Alert { at: Local::now(), backend: id.clone(), event });
        }
        assert_eq!(state.alerts.len(), MAX_ALERTS);
        assert_eq!(state.alert_counts[&(id, "kv_cache_high")], MAX_ALERTS as u64 + 5);
    }
}
