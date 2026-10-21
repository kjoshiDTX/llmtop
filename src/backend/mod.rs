//! The `Backend` trait and the pieces adapters share.

mod llamacpp;
mod mini_engine;
mod ollama;
mod vllm;

pub use llamacpp::LlamaCpp;
pub use mini_engine::MiniEngine;
pub use ollama::Ollama;
pub use vllm::Vllm;

use crate::snapshot::Snapshot;
use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::time::{Duration, Instant};

/// Which backend a value came from: its kind and the host it runs on. Kept
/// as two fields, not one string, so the exporter can emit them as separate
/// Prometheus labels. Deriving `Ord` sorts by kind, then instance.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackendId {
    pub kind: &'static str,
    pub instance: String,
}

/// Displays as `vllm localhost:8000`, the label used in the table and in recordings.
impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.kind, self.instance)
    }
}

impl BackendId {
    /// Normalizes `url` and returns it with the id built from its host.
    fn from_url(kind: &'static str, url: &str) -> (String, Self) {
        let (base_url, instance) = normalize_url(url);
        (base_url, Self { kind, instance })
    }
}

/// One LLM runtime llmtop can watch.
///
/// A poll is two steps: `fetch` does the network I/O and returns the raw
/// responses, `parse` turns them into a `Snapshot`. Keeping them apart lets
/// `--record` save exactly what the server sent, and lets tests replay those
/// recordings through the same `parse` with no network.
pub trait Backend: Send {
    /// What one `fetch` returns. An associated type: each adapter picks its
    /// own (a single response body, or several), and the trait's users can
    /// still name it as `B::Raw`. The bounds let it be written to and read
    /// back from a recording.
    type Raw: Serialize + DeserializeOwned + Send;

    /// The returned reference borrows from `self`.
    fn id(&self) -> &BackendId;

    /// Written as "returns a future that is `Send`" rather than as an
    /// `async fn`, because each backend runs in its own `tokio::spawn`ed task
    /// and tokio may move tasks between threads. Implementations can still use
    /// `async fn`; the compiler checks that the future they produce is `Send`.
    fn fetch(&mut self) -> impl Future<Output = Result<Self::Raw>> + Send;

    /// Takes `&mut self` so adapters can keep state between polls, such as the
    /// previous counter values used to compute tokens per second. `now` is
    /// passed in rather than read from the clock so replays control time.
    fn parse(&mut self, raw: &Self::Raw, now: Instant) -> Result<Snapshot>;
}

/// One client shared by every backend. `reqwest::Client` is a handle to a
/// connection pool behind an `Arc`, so cloning it is cheap and shares the pool.
pub fn http_client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder().timeout(Duration::from_secs(3)).build()?)
}

async fn get_text(http: &reqwest::Client, url: &str) -> Result<String> {
    let resp = http.get(url).send().await.with_context(|| format!("could not reach {url}"))?;
    Ok(resp.error_for_status()?.text().await?)
}

/// Turns `http://localhost:8000/` into `("http://localhost:8000", "localhost:8000")`.
fn normalize_url(url: &str) -> (String, String) {
    let url = url.trim_end_matches('/');
    let url = if url.contains("://") { url.to_string() } else { format!("http://{url}") };
    let host = url.split_once("://").map_or(url.as_str(), |(_, h)| h).to_string();
    (url, host)
}

/// Turns ever-increasing counters, such as tokens generated, into per-second
/// rates by remembering the previous value of each one.
#[derive(Default)]
pub struct CounterRate {
    prev: HashMap<String, (f64, Instant)>,
}

impl CounterRate {
    /// Returns `None` on the first sample, and after a counter goes backwards,
    /// which means the server restarted.
    pub fn update(&mut self, key: &str, value: f64, now: Instant) -> Option<f64> {
        let (prev_value, prev_at) = self.prev.insert(key.to_string(), (value, now))?;
        let secs = now.checked_duration_since(prev_at)?.as_secs_f64();
        if secs <= 0.0 || value < prev_value {
            return None;
        }
        Some((value - prev_value) / secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_urls() {
        assert_eq!(
            normalize_url("localhost:8000/"),
            ("http://localhost:8000".into(), "localhost:8000".into())
        );
        assert_eq!(normalize_url("https://gpu1:8000").1, "gpu1:8000");
    }

    #[test]
    fn rate_needs_two_samples_and_survives_resets() {
        let mut r = CounterRate::default();
        let t0 = Instant::now();
        assert_eq!(r.update("tok", 100.0, t0), None);
        assert_eq!(r.update("tok", 300.0, t0 + Duration::from_secs(2)), Some(100.0));
        assert_eq!(r.update("tok", 5.0, t0 + Duration::from_secs(3)), None);
        assert_eq!(r.update("tok", 15.0, t0 + Duration::from_secs(4)), Some(10.0));
    }
}
