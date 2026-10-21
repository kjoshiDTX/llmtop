//! The llama.cpp server, via `/metrics` (needs `--metrics`) and `/v1/models`.
//!
//! Its metrics carry no model label, so the name comes from `/v1/models`,
//! fetched once. Current releases no longer export KV-cache usage; the old
//! `kv_cache_usage_ratio` gauge is still read if a server sends it.

use super::{normalize_url, Backend, CounterRate};
use crate::prom;
use crate::snapshot::{ModelStats, Snapshot};
use anyhow::{bail, Context, Result};
use reqwest::StatusCode;
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct ModelList {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

pub struct LlamaCpp {
    name: String,
    base_url: String,
    http: reqwest::Client,
    model: Option<String>,
    rates: CounterRate,
}

impl LlamaCpp {
    pub fn new(url: &str, http: reqwest::Client) -> Self {
        let (base_url, host) = normalize_url(url);
        Self {
            name: format!("llama.cpp {host}"),
            base_url,
            http,
            model: None,
            rates: CounterRate::default(),
        }
    }

    async fn fetch_metrics(&self) -> Result<String> {
        let url = format!("{}/metrics", self.base_url);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("could not reach {url}"))?;
        if resp.status() == StatusCode::NOT_IMPLEMENTED {
            bail!("metrics are off; start llama-server with --metrics");
        }
        Ok(resp.error_for_status()?.text().await?)
    }

    async fn fetch_model_name(&self) -> Result<String> {
        let list: ModelList = self
            .http
            .get(format!("{}/v1/models", self.base_url))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        list.data.into_iter().next().map(|m| m.id).context("server lists no models")
    }
}

impl Backend for LlamaCpp {
    fn name(&self) -> &str {
        &self.name
    }

    async fn poll(&mut self) -> Result<Snapshot> {
        let text = if self.model.is_some() {
            self.fetch_metrics().await?
        } else {
            // First poll: fetch both at once. The name is optional, so a
            // failure there falls back to a placeholder and is retried later.
            let (text, name) = tokio::join!(self.fetch_metrics(), self.fetch_model_name());
            self.model = name.ok();
            text?
        };
        let model = self.model.as_deref().unwrap_or("(unknown model)");
        Ok(snapshot_from_metrics(model, &text, &mut self.rates, Instant::now()))
    }
}

/// Throughput comes from decode steps, not `tokens_predicted_total`: that
/// counter only moves when a request finishes, so it reads 0 mid-generation.
/// Each decode step produces one token per busy slot, so tokens/s is about
/// decodes/s times `n_busy_slots_per_decode`. That gauge is a lifetime
/// average, and prompt batches count as decodes too, so this is an estimate;
/// it is exact for a single slot generating text.
fn snapshot_from_metrics(model: &str, text: &str, rates: &mut CounterRate, now: Instant) -> Snapshot {
    let mut stats = ModelStats::new(model);
    let mut decodes = None;
    let mut slots_per_decode = 1.0;
    for s in prom::parse(text) {
        match s.name {
            "llamacpp:requests_processing" => stats.requests_running = Some(s.value as u64),
            "llamacpp:requests_deferred" => stats.requests_waiting = Some(s.value as u64),
            "llamacpp:kv_cache_usage_ratio" => stats.kv_cache_usage = Some(s.value),
            "llamacpp:n_decode_total" => decodes = Some(s.value),
            "llamacpp:n_busy_slots_per_decode" if s.value > 0.0 => slots_per_decode = s.value,
            _ => {}
        }
    }
    stats.gen_tokens_per_sec = decodes
        .and_then(|d| rates.update("decodes", d, now))
        .map(|per_sec| per_sec * slots_per_decode);
    Snapshot { models: vec![stats] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // Recorded from llama-server 0.6.0 (build 11429) after one 20-token request.
    const METRICS: &str = include_str!("../../fixtures/llamacpp_metrics.txt");
    const MODELS: &str = include_str!("../../fixtures/llamacpp_models.json");

    #[test]
    fn reads_recorded_metrics() {
        let snap = snapshot_from_metrics("qwen", METRICS, &mut CounterRate::default(), Instant::now());
        let m = &snap.models[0];
        assert_eq!(m.requests_running, Some(0));
        assert_eq!(m.requests_waiting, Some(0));
        assert_eq!(m.kv_cache_usage, None, "0.6.0 doesn't export KV usage");
    }

    #[test]
    fn computes_tokens_per_second() {
        let mut rates = CounterRate::default();
        let t0 = Instant::now();
        snapshot_from_metrics("qwen", METRICS, &mut rates, t0);
        // 100 more decode steps in 2 s, with 2 busy slots per step.
        let later = METRICS
            .replace("n_decode_total 20", "n_decode_total 120")
            .replace("n_busy_slots_per_decode 1", "n_busy_slots_per_decode 2");
        let snap = snapshot_from_metrics("qwen", &later, &mut rates, t0 + Duration::from_secs(2));
        assert_eq!(snap.models[0].gen_tokens_per_sec, Some(100.0));
    }

    #[test]
    fn reads_model_name_from_recorded_list() {
        let list: ModelList = serde_json::from_str(MODELS).unwrap();
        assert_eq!(list.data[0].id, "qwen2.5-0.5b");
    }
}
