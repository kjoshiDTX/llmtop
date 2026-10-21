//! Turns a stream of snapshots from one backend into state-change events:
//! a model falling back to CPU, the KV cache filling up, a model being
//! swapped out, the backend going down.
//!
//! Each condition is a small state machine: an enum holding the current
//! state, and a `match` on (current state, new reading) that decides the next
//! state and whether the change is worth an event. Events fire on
//! transitions only, so a model that stays on CPU is reported once.

use crate::snapshot::{ModelStats, Placement, Snapshot};
use anyhow::Result;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct Thresholds {
    /// KV usage at or above this raises an alert...
    pub kv_high: f64,
    /// ...which clears only once usage drops below this. The gap between the
    /// two (hysteresis) stops a cache hovering at 90% from flapping.
    pub kv_clear: f64,
    /// A model that disappears and is replaced within this window counts as
    /// a swap rather than an unload followed by a load.
    pub swap_window: Duration,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self { kv_high: 0.90, kv_clear: 0.80, swap_window: Duration::from_secs(30) }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    BackendDown { error: String },
    BackendRecovered,
    ModelLoaded { model: String },
    ModelUnloaded { model: String },
    /// The same model went away and came back, e.g. reloaded with new options.
    ModelReloaded { model: String },
    ModelSwapped { from: String, to: String },
    /// `gpu_share` is 0.0 to 1.0; 0.0 means entirely on CPU.
    FellBackToCpu { model: String, gpu_share: f64 },
    BackOnGpu { model: String },
    KvCacheHigh { model: String, usage: f64 },
    KvCacheRecovered { model: String, usage: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Severity {
    Warning,
    Info,
}

impl Event {
    /// Every value `name` can return. The exporter starts each counter at 0,
    /// because Prometheus's `increase()` can't see a series' first increment
    /// if the series only appears when it reaches 1.
    pub const NAMES: [&'static str; 10] = [
        "backend_down",
        "backend_recovered",
        "model_loaded",
        "model_unloaded",
        "model_reloaded",
        "model_swapped",
        "fell_back_to_cpu",
        "back_on_gpu",
        "kv_cache_high",
        "kv_cache_recovered",
    ];

    /// A stable snake_case name, used as the `alert` label in exported metrics.
    pub fn name(&self) -> &'static str {
        match self {
            Event::BackendDown { .. } => "backend_down",
            Event::BackendRecovered => "backend_recovered",
            Event::ModelLoaded { .. } => "model_loaded",
            Event::ModelUnloaded { .. } => "model_unloaded",
            Event::ModelReloaded { .. } => "model_reloaded",
            Event::ModelSwapped { .. } => "model_swapped",
            Event::FellBackToCpu { .. } => "fell_back_to_cpu",
            Event::BackOnGpu { .. } => "back_on_gpu",
            Event::KvCacheHigh { .. } => "kv_cache_high",
            Event::KvCacheRecovered { .. } => "kv_cache_recovered",
        }
    }

    pub fn severity(&self) -> Severity {
        match self {
            Event::BackendDown { .. }
            | Event::ModelSwapped { .. }
            | Event::FellBackToCpu { .. }
            | Event::KvCacheHigh { .. } => Severity::Warning,
            _ => Severity::Info,
        }
    }
}

/// The plain-language alert text.
impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Event::BackendDown { error } => write!(f, "unreachable: {error}"),
            Event::BackendRecovered => write!(f, "reachable again"),
            Event::ModelLoaded { model } => write!(f, "{model} loaded"),
            Event::ModelUnloaded { model } => write!(f, "{model} unloaded"),
            Event::ModelReloaded { model } => write!(f, "{model} reloaded from disk"),
            Event::ModelSwapped { from, to } => write!(
                f,
                "{from} was swapped out for {to}. Each swap reloads weights from disk; \
                 if this keeps happening, the models don't fit in memory together"
            ),
            Event::FellBackToCpu { model, gpu_share } if *gpu_share == 0.0 => {
                write!(f, "{model} is running entirely on CPU; generation will be much slower")
            }
            Event::FellBackToCpu { model, gpu_share } => write!(
                f,
                "{model} is running partly on CPU ({:.0}% on GPU); generation will be much slower",
                gpu_share * 100.0
            ),
            Event::BackOnGpu { model } => write!(f, "{model} is fully on GPU again"),
            Event::KvCacheHigh { model, usage } => write!(
                f,
                "{model} KV cache is {:.0}% full; new requests may queue or be preempted",
                usage * 100.0
            ),
            Event::KvCacheRecovered { model, usage } => {
                write!(f, "{model} KV cache is back down to {:.0}%", usage * 100.0)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Health {
    Unknown,
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
enum Residency {
    #[default]
    Gpu,
    PartlyCpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
enum Kv {
    #[default]
    Normal,
    High,
}

/// Starts as "on GPU, KV normal", so a model first seen on CPU or with a
/// full cache raises an alert straight away.
#[derive(Debug, Default)]
struct ModelState {
    residency: Residency,
    kv: Kv,
}

impl ModelState {
    fn step(&mut self, m: &ModelStats, th: &Thresholds, events: &mut Vec<Event>) {
        if let Some(placement) = m.placement {
            self.residency = match (self.residency, placement) {
                (Residency::Gpu, Placement::Split(gpu_share)) => {
                    events.push(Event::FellBackToCpu { model: m.name.clone(), gpu_share });
                    Residency::PartlyCpu
                }
                (Residency::Gpu, Placement::Cpu) => {
                    events.push(Event::FellBackToCpu { model: m.name.clone(), gpu_share: 0.0 });
                    Residency::PartlyCpu
                }
                (Residency::PartlyCpu, Placement::Gpu) => {
                    events.push(Event::BackOnGpu { model: m.name.clone() });
                    Residency::Gpu
                }
                (unchanged, _) => unchanged,
            };
        }

        if let Some(usage) = m.kv_cache_usage {
            self.kv = match self.kv {
                Kv::Normal if usage >= th.kv_high => {
                    events.push(Event::KvCacheHigh { model: m.name.clone(), usage });
                    Kv::High
                }
                Kv::High if usage < th.kv_clear => {
                    events.push(Event::KvCacheRecovered { model: m.name.clone(), usage });
                    Kv::Normal
                }
                unchanged => unchanged,
            };
        }
    }
}

/// Tracks one backend. Owned by that backend's polling task, so it needs no lock.
pub struct Detector {
    th: Thresholds,
    health: Health,
    /// `None` until the first successful poll. Models already loaded at that
    /// point are the starting state, not "loaded" events.
    models: Option<BTreeMap<String, ModelState>>,
    /// Models that disappeared recently, with when. Reported as unloaded only
    /// once the swap window passes without a replacement arriving.
    departed: Vec<(String, Instant)>,
}

impl Detector {
    pub fn new(th: Thresholds) -> Self {
        Self { th, health: Health::Unknown, models: None, departed: Vec::new() }
    }

    pub fn observe(&mut self, result: &Result<Snapshot>, now: Instant) -> Vec<Event> {
        let mut events = Vec::new();

        self.health = match (self.health, result) {
            (Health::Unknown | Health::Up, Err(err)) => {
                events.push(Event::BackendDown { error: format!("{err:#}") });
                Health::Down
            }
            (Health::Down, Ok(_)) => {
                events.push(Event::BackendRecovered);
                Health::Up
            }
            (Health::Down, Err(_)) => Health::Down,
            (Health::Unknown | Health::Up, Ok(_)) => Health::Up,
        };

        self.expire_departures(now, &mut events);
        if let Ok(snapshot) = result {
            self.observe_models(snapshot, now, &mut events);
        }
        events
    }

    fn observe_models(&mut self, snapshot: &Snapshot, now: Instant, events: &mut Vec<Event>) {
        let first_poll = self.models.is_none();
        // Borrows only the `models` field, so `self.departed` and `self.th`
        // stay usable below.
        let models = self.models.get_or_insert_with(BTreeMap::new);

        let current: BTreeSet<&str> = snapshot.models.iter().map(|m| m.name.as_str()).collect();
        let gone: Vec<String> =
            models.keys().filter(|name| !current.contains(name.as_str())).cloned().collect();
        for name in gone {
            models.remove(&name);
            self.departed.push((name, now));
        }

        for m in &snapshot.models {
            if !models.contains_key(&m.name) {
                if !first_poll {
                    events.push(arrival(&mut self.departed, &m.name));
                }
                models.insert(m.name.clone(), ModelState::default());
            }
            let state = models.get_mut(&m.name).expect("inserted above");
            state.step(m, &self.th, events);
        }
    }

    fn expire_departures(&mut self, now: Instant, events: &mut Vec<Event>) {
        let window = self.th.swap_window;
        // `retain` keeps departures still inside the window and reports the rest.
        self.departed.retain(|(name, at)| {
            let expired = now.saturating_duration_since(*at) > window;
            if expired {
                events.push(Event::ModelUnloaded { model: name.clone() });
            }
            !expired
        });
    }
}

/// Decides what a newly seen model means, given what left recently. A free
/// function rather than a method so it can borrow `departed` mutably while
/// the caller still holds a borrow of the `models` map.
fn arrival(departed: &mut Vec<(String, Instant)>, name: &str) -> Event {
    if let Some(i) = departed.iter().position(|(n, _)| n == name) {
        departed.remove(i);
        return Event::ModelReloaded { model: name.to_string() };
    }
    if !departed.is_empty() {
        let (from, _) = departed.remove(0);
        return Event::ModelSwapped { from, to: name.to_string() };
    }
    Event::ModelLoaded { model: name.to_string() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    fn snap(models: &[ModelStats]) -> Result<Snapshot> {
        Ok(Snapshot { models: models.to_vec() })
    }

    fn kv(name: &str, usage: f64) -> ModelStats {
        ModelStats { kv_cache_usage: Some(usage), ..ModelStats::new(name) }
    }

    fn placed(name: &str, p: Placement) -> ModelStats {
        ModelStats { placement: Some(p), ..ModelStats::new(name) }
    }

    /// Feeds results one second apart and returns every event, in order.
    fn run(results: Vec<Result<Snapshot>>) -> Vec<Event> {
        let mut d = Detector::new(Thresholds::default());
        let t0 = Instant::now();
        let mut events = Vec::new();
        for (i, r) in results.iter().enumerate() {
            events.extend(d.observe(r, t0 + Duration::from_secs(i as u64)));
        }
        events
    }

    #[test]
    fn kv_alert_has_hysteresis() {
        let usages = [0.50, 0.91, 0.85, 0.93, 0.81, 0.79, 0.95];
        let events = run(usages.iter().map(|u| snap(&[kv("m", *u)])).collect());
        assert_eq!(
            events,
            vec![
                Event::KvCacheHigh { model: "m".into(), usage: 0.91 },
                Event::KvCacheRecovered { model: "m".into(), usage: 0.79 },
                Event::KvCacheHigh { model: "m".into(), usage: 0.95 },
            ]
        );
    }

    #[test]
    fn placement_changes_fire_once() {
        let events = run(vec![
            snap(&[placed("m", Placement::Gpu)]),
            snap(&[placed("m", Placement::Split(0.6))]),
            snap(&[placed("m", Placement::Cpu)]),
            snap(&[placed("m", Placement::Gpu)]),
        ]);
        assert_eq!(
            events,
            vec![
                Event::FellBackToCpu { model: "m".into(), gpu_share: 0.6 },
                Event::BackOnGpu { model: "m".into() },
            ]
        );
    }

    #[test]
    fn model_already_on_cpu_at_startup_is_reported() {
        let events = run(vec![snap(&[placed("m", Placement::Cpu)])]);
        assert_eq!(events, vec![Event::FellBackToCpu { model: "m".into(), gpu_share: 0.0 }]);
    }

    #[test]
    fn swap_across_an_empty_poll() {
        let a = ModelStats::new("a");
        let b = ModelStats::new("b");
        let events = run(vec![snap(&[a]), snap(&[]), snap(&[b])]);
        assert_eq!(events, vec![Event::ModelSwapped { from: "a".into(), to: "b".into() }]);
    }

    #[test]
    fn unload_reported_after_window_and_load_without_departure() {
        let mut d = Detector::new(Thresholds::default());
        let t0 = Instant::now();
        assert_eq!(d.observe(&snap(&[ModelStats::new("a")]), t0), vec![]);
        assert_eq!(d.observe(&snap(&[]), t0 + Duration::from_secs(1)), vec![]);
        assert_eq!(
            d.observe(&snap(&[]), t0 + Duration::from_secs(40)),
            vec![Event::ModelUnloaded { model: "a".into() }]
        );
        assert_eq!(
            d.observe(&snap(&[ModelStats::new("b")]), t0 + Duration::from_secs(41)),
            vec![Event::ModelLoaded { model: "b".into() }]
        );
    }

    #[test]
    fn backend_down_once_then_recovered() {
        let events = run(vec![
            snap(&[]),
            Err(anyhow!("connection refused")),
            Err(anyhow!("connection refused")),
            snap(&[]),
        ]);
        assert_eq!(
            events,
            vec![Event::BackendDown { error: "connection refused".into() }, Event::BackendRecovered]
        );
    }

    #[test]
    fn names_lists_every_variant_once() {
        let model = || "m".to_string();
        let all = [
            Event::BackendDown { error: "e".into() },
            Event::BackendRecovered,
            Event::ModelLoaded { model: model() },
            Event::ModelUnloaded { model: model() },
            Event::ModelReloaded { model: model() },
            Event::ModelSwapped { from: model(), to: model() },
            Event::FellBackToCpu { model: model(), gpu_share: 0.0 },
            Event::BackOnGpu { model: model() },
            Event::KvCacheHigh { model: model(), usage: 1.0 },
            Event::KvCacheRecovered { model: model(), usage: 0.0 },
        ];
        let names: Vec<_> = all.iter().map(Event::name).collect();
        assert_eq!(names, Event::NAMES);
    }

    #[test]
    fn plain_language_messages() {
        let e = Event::FellBackToCpu { model: "qwen".into(), gpu_share: 0.4 };
        assert_eq!(e.to_string(), "qwen is running partly on CPU (40% on GPU); generation will be much slower");
        assert_eq!(e.severity(), Severity::Warning);
        assert_eq!(Event::BackOnGpu { model: "qwen".into() }.severity(), Severity::Info);
    }
}
