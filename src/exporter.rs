//! `--listen ADDR`: serves the shared state as Prometheus metrics on
//! `/metrics`, using axum.
//!
//! Every backend kind maps into the same metric names, so one Grafana query
//! covers Ollama, vLLM, llama.cpp and mini-engine workers alike. Each sample
//! carries `kind` and `instance` labels, plus `model` where it applies.

use crate::backend::BackendId;
use crate::snapshot::{ModelStats, Placement};
use crate::state::{BackendStatus, Shared, State};
use axum::extract::State as AxumState;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use std::fmt::Write;
use std::sync::Arc;
use tokio::net::TcpListener;

/// Runs until the process exits. The listener is bound by the caller, so a
/// port that's already taken is reported before polling starts.
pub async fn serve(listener: TcpListener, shared: Arc<Shared>) -> std::io::Result<()> {
    let app = Router::new()
        .route("/", get(|| async { "llmtop exporter. Metrics are at /metrics.\n" }))
        .route("/metrics", get(metrics))
        // Handlers receive this through the `State` extractor below.
        .with_state(shared);
    axum::serve(listener, app).await
}

/// axum builds the arguments from the request: `AxumState` pulls out the
/// `Arc<Shared>` given to `with_state`. Anything that implements
/// `IntoResponse` can be returned, here a (headers, body) tuple.
async fn metrics(AxumState(shared): AxumState<Arc<Shared>>) -> impl IntoResponse {
    // `encode` runs while holding the read lock and returns an owned String,
    // so the lock is released before the response is sent.
    let body = encode(&shared.state.read().expect("state lock poisoned"));
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")], body)
}

/// One gauge per `ModelStats` field: its name, help text, and how to read it.
/// A plain `fn` pointer is enough, since none of these capture anything.
type ModelGauge = (&'static str, &'static str, fn(&ModelStats) -> Option<f64>);

const MODEL_GAUGES: [ModelGauge; 6] = [
    ("llmtop_model_size_bytes", "Memory the model occupies, GPU and system combined.", |m| {
        m.size_bytes.map(|b| b as f64)
    }),
    (
        "llmtop_model_gpu_ratio",
        "Share of the model's weights in GPU memory. Below 1 means part runs on CPU.",
        |m| m.placement.map(gpu_ratio),
    ),
    ("llmtop_kv_cache_usage_ratio", "Share of the KV cache in use, 0 to 1.", |m| m.kv_cache_usage),
    ("llmtop_requests_running", "Requests currently generating.", |m| {
        m.requests_running.map(|n| n as f64)
    }),
    ("llmtop_requests_waiting", "Requests queued, waiting to start.", |m| {
        m.requests_waiting.map(|n| n as f64)
    }),
    (
        "llmtop_generation_tokens_per_second",
        "Generated tokens per second since the previous poll.",
        |m| m.gen_tokens_per_sec,
    ),
];

fn gpu_ratio(p: Placement) -> f64 {
    match p {
        Placement::Gpu => 1.0,
        Placement::Cpu => 0.0,
        Placement::Split(share) => share,
    }
}

pub fn encode(state: &State) -> String {
    let mut w = Writer::default();

    w.family("llmtop_backend_up", "gauge", "1 if the last poll of this backend succeeded.");
    for (id, status) in &state.backends {
        let up = match status {
            BackendStatus::Pending => continue,
            BackendStatus::Up(_) => 1.0,
            BackendStatus::Down(_) => 0.0,
        };
        w.sample("llmtop_backend_up", &backend_labels(id), up);
    }

    // Only backends that answered have models to report.
    let models: Vec<(&BackendId, &ModelStats)> = state
        .backends
        .iter()
        .filter_map(|(id, status)| match status {
            BackendStatus::Up(snap) => Some((id, snap)),
            _ => None,
        })
        .flat_map(|(id, snap)| snap.models.iter().map(move |m| (id, m)))
        .collect();

    w.family("llmtop_model_loaded", "gauge", "1 for each model a backend reports as loaded.");
    for (id, m) in &models {
        w.sample("llmtop_model_loaded", &model_labels(id, m), 1.0);
    }
    for (name, help, read) in MODEL_GAUGES {
        w.family(name, "gauge", help);
        for (id, m) in &models {
            if let Some(value) = read(m) {
                w.sample(name, &model_labels(id, m), value);
            }
        }
    }

    w.family("llmtop_alerts_total", "counter", "State-change alerts raised since llmtop started.");
    for ((id, alert), count) in &state.alert_counts {
        let mut labels = backend_labels(id);
        labels.push(("alert", alert));
        w.sample("llmtop_alerts_total", &labels, *count as f64);
    }

    if let Some(sys) = &state.system {
        w.family("llmtop_host_cpu_usage_ratio", "gauge", "Host CPU usage across all cores, 0 to 1.");
        w.sample("llmtop_host_cpu_usage_ratio", &[], f64::from(sys.cpu_percent) / 100.0);
        w.family("llmtop_host_memory_used_bytes", "gauge", "Host RAM in use.");
        w.sample("llmtop_host_memory_used_bytes", &[], sys.mem_used as f64);
        w.family("llmtop_host_memory_total_bytes", "gauge", "Host RAM installed.");
        w.sample("llmtop_host_memory_total_bytes", &[], sys.mem_total as f64);
    }
    w.out
}

/// Label pairs borrow from the id and model they describe, so building them
/// copies no strings.
fn backend_labels(id: &BackendId) -> Vec<(&str, &str)> {
    vec![("kind", id.kind), ("instance", &id.instance)]
}

fn model_labels<'a>(id: &'a BackendId, m: &'a ModelStats) -> Vec<(&'a str, &'a str)> {
    let mut labels = backend_labels(id);
    labels.push(("model", &m.name));
    labels
}

/// Writes the Prometheus text exposition format.
#[derive(Default)]
struct Writer {
    out: String,
}

impl Writer {
    fn family(&mut self, name: &str, kind: &str, help: &str) {
        let _ = writeln!(self.out, "# HELP {name} {help}\n# TYPE {name} {kind}");
    }

    fn sample(&mut self, name: &str, labels: &[(&str, &str)], value: f64) {
        self.out.push_str(name);
        if !labels.is_empty() {
            self.out.push('{');
            for (i, (key, value)) in labels.iter().enumerate() {
                if i > 0 {
                    self.out.push(',');
                }
                let _ = write!(self.out, "{key}=\"{}\"", escape(value));
            }
            self.out.push('}');
        }
        let _ = writeln!(self.out, " {}", format_value(value));
    }
}

/// Label values escape backslash, double quote and newline.
fn escape(value: &str) -> String {
    value.replace('\\', r"\\").replace('"', "\\\"").replace('\n', r"\n")
}

/// Rust prints infinity as `inf`; Prometheus expects `+Inf`.
fn format_value(v: f64) -> String {
    if v.is_nan() {
        "NaN".into()
    } else if v.is_infinite() {
        if v > 0.0 { "+Inf".into() } else { "-Inf".into() }
    } else {
        v.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prom;
    use crate::snapshot::Snapshot;
    use crate::system::SystemStats;

    fn id(kind: &'static str, instance: &str) -> BackendId {
        BackendId { kind, instance: instance.into() }
    }

    fn sample_state() -> State {
        let mut state = State {
            system: Some(SystemStats { cpu_percent: 25.0, mem_used: 8, mem_total: 16 }),
            ..State::default()
        };
        let ollama = ModelStats {
            size_bytes: Some(1000),
            placement: Some(Placement::Split(0.75)),
            ..ModelStats::new("qwen2.5:0.5b")
        };
        state.backends.insert(id("ollama", "localhost:11434"), BackendStatus::Up(Snapshot { models: vec![ollama] }));
        let worker = ModelStats {
            kv_cache_usage: Some(0.5),
            requests_running: Some(3),
            gen_tokens_per_sec: Some(42.5),
            ..ModelStats::new(r#"odd "name" \ here"#)
        };
        state.backends.insert(id("mini-engine", "localhost:8001"), BackendStatus::Up(Snapshot { models: vec![worker] }));
        state.backends.insert(id("vllm", "gpu1:8000"), BackendStatus::Down("refused".into()));
        state.backends.insert(id("llama.cpp", "localhost:8089"), BackendStatus::Pending);
        state.alert_counts.insert((id("ollama", "localhost:11434"), "fell_back_to_cpu"), 2);
        state
    }

    /// Finds the value of the sample with this name whose labels include all of `want`.
    fn value(samples: &[prom::Sample], name: &str, want: &[(&str, &str)]) -> Option<f64> {
        samples
            .iter()
            .find(|s| s.name == name && want.iter().all(|(k, v)| s.label(k) == Some(v)))
            .map(|s| s.value)
    }

    #[test]
    fn round_trips_through_the_parser() {
        let text = encode(&sample_state());
        let samples = prom::parse(&text);

        let ollama = [("kind", "ollama"), ("instance", "localhost:11434")];
        assert_eq!(value(&samples, "llmtop_backend_up", &ollama), Some(1.0));
        assert_eq!(value(&samples, "llmtop_backend_up", &[("kind", "vllm")]), Some(0.0));
        assert_eq!(value(&samples, "llmtop_backend_up", &[("kind", "llama.cpp")]), None, "pending");
        assert_eq!(value(&samples, "llmtop_model_gpu_ratio", &ollama), Some(0.75));
        assert_eq!(value(&samples, "llmtop_model_size_bytes", &ollama), Some(1000.0));
        assert_eq!(value(&samples, "llmtop_kv_cache_usage_ratio", &ollama), None, "not reported");

        let worker = [("model", r#"odd "name" \ here"#)];
        assert_eq!(value(&samples, "llmtop_kv_cache_usage_ratio", &worker), Some(0.5));
        assert_eq!(value(&samples, "llmtop_generation_tokens_per_second", &worker), Some(42.5));

        let alert = [("kind", "ollama"), ("alert", "fell_back_to_cpu")];
        assert_eq!(value(&samples, "llmtop_alerts_total", &alert), Some(2.0));
        assert_eq!(value(&samples, "llmtop_host_cpu_usage_ratio", &[]), Some(0.25));
    }

    #[test]
    fn each_family_has_help_and_type_once() {
        let text = encode(&sample_state());
        for (name, ..) in MODEL_GAUGES {
            assert_eq!(text.matches(&format!("# TYPE {name} gauge\n")).count(), 1);
        }
        assert!(text.contains("# TYPE llmtop_alerts_total counter\n"));
        assert!(text.contains(r#"model="odd \"name\" \\ here""#));
    }

    #[test]
    fn formats_special_values() {
        assert_eq!(format_value(f64::INFINITY), "+Inf");
        assert_eq!(format_value(0.5), "0.5");
        assert_eq!(format_value(3.0), "3");
    }
}
