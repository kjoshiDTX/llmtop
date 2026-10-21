//! vLLM, via its Prometheus `/metrics` endpoint. Every series carries a
//! `model_name` label, and with data parallelism there is one series per
//! engine (the `engine` label), so values are combined per model.

use super::{normalize_url, Backend, CounterRate};
use crate::prom;
use crate::snapshot::{ModelStats, Snapshot};
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::time::Instant;

pub struct Vllm {
    name: String,
    base_url: String,
    http: reqwest::Client,
    rates: CounterRate,
}

impl Vllm {
    pub fn new(url: &str, http: reqwest::Client) -> Self {
        let (base_url, host) = normalize_url(url);
        Self { name: format!("vllm {host}"), base_url, http, rates: CounterRate::default() }
    }
}

impl Backend for Vllm {
    fn name(&self) -> &str {
        &self.name
    }

    async fn poll(&mut self) -> Result<Snapshot> {
        let url = format!("{}/metrics", self.base_url);
        let text = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("could not reach {url}"))?
            .error_for_status()?
            .text()
            .await?;
        Ok(snapshot_from_metrics(&text, &mut self.rates, Instant::now()))
    }
}

/// Kept separate from `poll` so tests can feed it recorded text and fake times.
fn snapshot_from_metrics(text: &str, rates: &mut CounterRate, now: Instant) -> Snapshot {
    let samples = prom::parse(text);

    // Keys borrow from `samples`, so this map can't outlive it. A BTreeMap
    // keeps models in a stable, sorted order between polls.
    let mut models: BTreeMap<&str, ModelStats> = BTreeMap::new();
    let mut generated: BTreeMap<&str, f64> = BTreeMap::new();

    for s in &samples {
        let Some(model) = s.label("model_name") else { continue };
        let stats = models.entry(model).or_insert_with(|| ModelStats::new(model));
        match s.name {
            "vllm:num_requests_running" => add(&mut stats.requests_running, s.value),
            "vllm:num_requests_waiting" => add(&mut stats.requests_waiting, s.value),
            // `gpu_cache_usage_perc` is the name used by older vLLM releases.
            "vllm:kv_cache_usage_perc" | "vllm:gpu_cache_usage_perc" => {
                let worst = stats.kv_cache_usage.unwrap_or(0.0).max(s.value);
                stats.kv_cache_usage = Some(worst);
            }
            "vllm:generation_tokens_total" => *generated.entry(model).or_default() += s.value,
            _ => {}
        }
    }

    for (model, total) in generated {
        // `models` must contain `model`, since both maps are filled in the same loop.
        models.get_mut(model).unwrap().gen_tokens_per_sec = rates.update(model, total, now);
    }

    Snapshot { models: models.into_values().collect() }
}

/// Gauges arrive as floats; counts of requests are whole numbers.
fn add(field: &mut Option<u64>, value: f64) {
    *field = Some(field.unwrap_or(0) + value.max(0.0) as u64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const METRICS: &str = include_str!("../../fixtures/vllm_metrics.txt");

    #[test]
    fn combines_engines_per_model() {
        let snap = snapshot_from_metrics(METRICS, &mut CounterRate::default(), Instant::now());
        assert_eq!(snap.models.len(), 1);
        let m = &snap.models[0];
        assert_eq!(m.name, "Qwen/Qwen2.5-7B-Instruct");
        assert_eq!(m.requests_running, Some(10));
        assert_eq!(m.requests_waiting, Some(2));
        assert_eq!(m.kv_cache_usage, Some(0.82));
        assert_eq!(m.size_bytes, None);
        assert_eq!(m.gen_tokens_per_sec, None, "first poll has no previous sample");
    }

    #[test]
    fn computes_tokens_per_second_across_polls() {
        let mut rates = CounterRate::default();
        let t0 = Instant::now();
        snapshot_from_metrics(METRICS, &mut rates, t0);

        // 2000 more tokens across both engines, 4 seconds later.
        let later = METRICS
            .replace("} 120000.0", "} 121500.0")
            .replace("} 80000.0", "} 80500.0");
        let snap = snapshot_from_metrics(&later, &mut rates, t0 + Duration::from_secs(4));
        assert_eq!(snap.models[0].gen_tokens_per_sec, Some(500.0));
    }

    #[test]
    fn accepts_old_kv_metric_name() {
        let text = r#"vllm:gpu_cache_usage_perc{model_name="m"} 0.5"#;
        let snap = snapshot_from_metrics(text, &mut CounterRate::default(), Instant::now());
        assert_eq!(snap.models[0].kv_cache_usage, Some(0.5));
    }
}
