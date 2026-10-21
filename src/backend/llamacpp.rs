//! The llama.cpp server, via `/metrics` (needs `--metrics`), `/slots` and
//! `/v1/models`.
//!
//! Its metrics carry no model label, so the name comes from `/v1/models`,
//! fetched once. Current releases no longer export KV-cache usage, so it is
//! computed from `/slots` instead; the old `kv_cache_usage_ratio` gauge is
//! used if a server still sends it and `/slots` is unavailable.

use super::{get_text, normalize_url, Backend, CounterRate};
use crate::prom;
use crate::snapshot::{ModelStats, Snapshot};
use anyhow::{bail, Context, Result};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Instant;

/// Everything one poll fetches, as the server sent it.
#[derive(Debug, Serialize, Deserialize)]
pub struct Raw {
    model: Option<String>,
    metrics: String,
    /// `None` if `/slots` is disabled (`--no-slots`) or failed.
    #[serde(default)]
    slots: Option<String>,
}

#[derive(Deserialize)]
struct ModelList {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

#[derive(Deserialize)]
struct Slot {
    n_ctx: u64,
    #[serde(default)]
    is_processing: bool,
    #[serde(default)]
    n_prompt_tokens: u64,
    /// Prompt tokens evaluated and reused from cache. Only in newer releases.
    n_prompt_tokens_processed: Option<u64>,
    #[serde(default)]
    n_prompt_tokens_cache: u64,
    /// An array in current releases, a single object in older ones.
    #[serde(default)]
    next_token: Value,
}

impl Slot {
    /// Tokens this slot holds in its KV cache. Idle slots may keep a cached
    /// prompt around, but it is reusable rather than pressure, so count 0.
    fn tokens_in_use(&self) -> u64 {
        if !self.is_processing {
            return 0;
        }
        let next = self.next_token.as_array().and_then(|a| a.first()).unwrap_or(&self.next_token);
        let decoded = next.get("n_decoded").and_then(Value::as_u64).unwrap_or(0);
        // In 0.6.0, `n_prompt_tokens` already includes generated tokens and
        // grows during generation, so the prompt length is rebuilt from the
        // processed and cached counts. Older servers lack those fields, and
        // their `n_prompt_tokens` is the prompt alone.
        let prompt = match self.n_prompt_tokens_processed {
            Some(processed) => processed + self.n_prompt_tokens_cache,
            None => self.n_prompt_tokens,
        };
        (prompt + decoded).min(self.n_ctx)
    }
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
        let body = get_text(&self.http, &format!("{}/v1/models", self.base_url)).await?;
        let list: ModelList = serde_json::from_str(&body)?;
        list.data.into_iter().next().map(|m| m.id).context("server lists no models")
    }
}

impl Backend for LlamaCpp {
    type Raw = Raw;

    fn name(&self) -> &str {
        &self.name
    }

    async fn fetch(&mut self) -> Result<Raw> {
        let need_name = self.model.is_none();
        // Bound to a variable so it outlives the futures that borrow it below.
        let slots_url = format!("{}/slots", self.base_url);
        // All three requests run at once. The model name is fetched until it
        // succeeds once; a failure there or on `/slots` isn't fatal.
        let (metrics, slots, name) = tokio::join!(
            self.fetch_metrics(),
            get_text(&self.http, &slots_url),
            async {
                if need_name { self.fetch_model_name().await.ok() } else { None }
            },
        );
        if name.is_some() {
            self.model = name;
        }
        Ok(Raw { model: self.model.clone(), metrics: metrics?, slots: slots.ok() })
    }

    fn parse(&mut self, raw: &Raw, now: Instant) -> Result<Snapshot> {
        let model = raw.model.as_deref().unwrap_or("(unknown model)");
        let mut stats = stats_from_metrics(model, &raw.metrics, &mut self.rates, now);
        if let Some(kv) = raw.slots.as_deref().and_then(kv_usage_from_slots) {
            stats.kv_cache_usage = Some(kv);
        }
        Ok(Snapshot { models: vec![stats] })
    }
}

/// Throughput comes from decode steps, not `tokens_predicted_total`: that
/// counter only moves when a request finishes, so it reads 0 mid-generation.
/// Each decode step produces one token per busy slot, so tokens/s is about
/// decodes/s times `n_busy_slots_per_decode`. That gauge is a lifetime
/// average, and prompt batches count as decodes too, so this is an estimate;
/// it is exact for a single slot generating text.
fn stats_from_metrics(model: &str, text: &str, rates: &mut CounterRate, now: Instant) -> ModelStats {
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
    stats
}

/// Share of all slots' context that running requests occupy.
fn kv_usage_from_slots(body: &str) -> Option<f64> {
    let slots: Vec<Slot> = serde_json::from_str(body).ok()?;
    let capacity: u64 = slots.iter().map(|s| s.n_ctx).sum();
    let used: u64 = slots.iter().map(Slot::tokens_in_use).sum();
    (capacity > 0).then(|| used as f64 / capacity as f64)
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
        let m = stats_from_metrics("qwen", METRICS, &mut CounterRate::default(), Instant::now());
        assert_eq!(m.requests_running, Some(0));
        assert_eq!(m.requests_waiting, Some(0));
        assert_eq!(m.kv_cache_usage, None, "0.6.0 doesn't export KV usage");
    }

    #[test]
    fn computes_tokens_per_second() {
        let mut rates = CounterRate::default();
        let t0 = Instant::now();
        stats_from_metrics("qwen", METRICS, &mut rates, t0);
        // 100 more decode steps in 2 s, with 2 busy slots per step.
        let later = METRICS
            .replace("n_decode_total 20", "n_decode_total 120")
            .replace("n_busy_slots_per_decode 1", "n_busy_slots_per_decode 2");
        let m = stats_from_metrics("qwen", &later, &mut rates, t0 + Duration::from_secs(2));
        assert_eq!(m.gen_tokens_per_sec, Some(100.0));
    }

    #[test]
    fn reads_model_name_from_recorded_list() {
        let list: ModelList = serde_json::from_str(MODELS).unwrap();
        assert_eq!(list.data[0].id, "qwen2.5-0.5b");
    }

    #[test]
    fn kv_usage_counts_only_busy_slots() {
        // Shapes as returned by llama-server 0.6.0: idle, then mid-generation.
        let idle = r#"[{"id":0,"n_ctx":512,"speculative":false,"is_processing":false}]"#;
        assert_eq!(kv_usage_from_slots(idle), Some(0.0));
        // `n_prompt_tokens` (114) already includes the 82 decoded tokens.
        let busy = r#"[
            {"id":0,"n_ctx":512,"is_processing":true,"n_prompt_tokens":114,
             "n_prompt_tokens_processed":32,"n_prompt_tokens_cache":0,
             "next_token":[{"has_next_token":true,"n_remain":318,"n_decoded":82}]},
            {"id":1,"n_ctx":512,"is_processing":false}
        ]"#;
        assert_eq!(kv_usage_from_slots(busy), Some(114.0 / 1024.0));
        // Older servers send `next_token` as an object.
        let old = r#"[{"n_ctx":100,"is_processing":true,"n_prompt_tokens":10,"next_token":{"n_decoded":40}}]"#;
        assert_eq!(kv_usage_from_slots(old), Some(0.5));
    }
}
