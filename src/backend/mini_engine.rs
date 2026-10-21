//! Workers from llm-inference-scheduler (`server/worker.py`), via `/metrics`.
//!
//! Each worker exports its engine's counters as `mini_engine_<key>` gauges
//! labelled with `worker_id`. The worker doesn't report its model name, so
//! the worker id stands in for it in the MODEL column.

use super::{get_text, Backend, BackendId, CounterRate};
use crate::prom;
use crate::snapshot::{ModelStats, Snapshot};
use anyhow::Result;
use std::collections::BTreeMap;
use std::time::Instant;

pub struct MiniEngine {
    id: BackendId,
    base_url: String,
    http: reqwest::Client,
    rates: CounterRate,
}

impl MiniEngine {
    pub fn new(url: &str, http: reqwest::Client) -> Self {
        let (base_url, id) = BackendId::from_url("mini-engine", url);
        Self { id, base_url, http, rates: CounterRate::default() }
    }
}

impl Backend for MiniEngine {
    /// The `/metrics` response body.
    type Raw = String;

    fn id(&self) -> &BackendId {
        &self.id
    }

    async fn fetch(&mut self) -> Result<String> {
        get_text(&self.http, &format!("{}/metrics", self.base_url)).await
    }

    fn parse(&mut self, raw: &String, now: Instant) -> Result<Snapshot> {
        Ok(snapshot_from_metrics(raw, &mut self.rates, now))
    }
}

/// The raw block counts for one worker, gathered before computing KV usage.
#[derive(Default)]
struct Blocks {
    total: Option<f64>,
    free: Option<f64>,
    in_prefix_cache: f64,
}

fn snapshot_from_metrics(text: &str, rates: &mut CounterRate, now: Instant) -> Snapshot {
    let samples = prom::parse(text);
    let mut workers: BTreeMap<&str, (ModelStats, Blocks)> = BTreeMap::new();

    for s in &samples {
        let Some(key) = s.name.strip_prefix("mini_engine_") else { continue };
        let worker = s.label("worker_id").unwrap_or("mini-engine");
        let (stats, blocks) =
            workers.entry(worker).or_insert_with(|| (ModelStats::new(worker), Blocks::default()));
        match key {
            "num_running" => stats.requests_running = Some(s.value as u64),
            "num_waiting" => stats.requests_waiting = Some(s.value as u64),
            "num_total_blocks" => blocks.total = Some(s.value),
            "num_free_blocks" => blocks.free = Some(s.value),
            "radix_tree_blocks" => blocks.in_prefix_cache = s.value,
            // Exported as a gauge, but it only ever counts up.
            "tokens_generated" => stats.gen_tokens_per_sec = rates.update(worker, s.value, now),
            _ => {}
        }
    }

    let models = workers
        .into_values()
        .map(|(mut stats, blocks)| {
            stats.kv_cache_usage = kv_usage(&blocks);
            stats
        })
        .collect();
    Snapshot { models }
}

/// Blocks that are neither free nor in the prefix cache, as a share of all
/// blocks. Prefix-cache blocks are left out because the engine evicts them
/// when it needs room, the same way vLLM's usage figure treats cached
/// blocks. The exception is cached prefixes that running requests share:
/// they are pinned but counted as free here, so this can read slightly low.
fn kv_usage(b: &Blocks) -> Option<f64> {
    let (total, free) = (b.total?, b.free?);
    (total > 0.0).then(|| ((total - free - b.in_prefix_cache) / total).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // The format `server/worker.py` writes: every key as a gauge with a `worker_id` label.
    const METRICS: &str = "\
# TYPE mini_engine_requests_finished gauge
mini_engine_requests_finished{worker_id=\"w1\"} 12
# TYPE mini_engine_tokens_generated gauge
mini_engine_tokens_generated{worker_id=\"w1\"} 3000
# TYPE mini_engine_num_waiting gauge
mini_engine_num_waiting{worker_id=\"w1\"} 3
# TYPE mini_engine_num_running gauge
mini_engine_num_running{worker_id=\"w1\"} 8
# TYPE mini_engine_num_free_blocks gauge
mini_engine_num_free_blocks{worker_id=\"w1\"} 200
# TYPE mini_engine_num_total_blocks gauge
mini_engine_num_total_blocks{worker_id=\"w1\"} 1024
# TYPE mini_engine_radix_tree_blocks gauge
mini_engine_radix_tree_blocks{worker_id=\"w1\"} 312
";

    #[test]
    fn maps_worker_metrics() {
        let mut rates = CounterRate::default();
        let t0 = Instant::now();
        let snap = snapshot_from_metrics(METRICS, &mut rates, t0);
        let m = &snap.models[0];
        assert_eq!(m.name, "w1");
        assert_eq!(m.requests_running, Some(8));
        assert_eq!(m.requests_waiting, Some(3));
        assert_eq!(m.kv_cache_usage, Some(512.0 / 1024.0));
        assert_eq!(m.gen_tokens_per_sec, None);

        let later = METRICS.replace("} 3000", "} 3120");
        let snap = snapshot_from_metrics(&later, &mut rates, t0 + Duration::from_secs(2));
        assert_eq!(snap.models[0].gen_tokens_per_sec, Some(60.0));
    }

    #[test]
    fn without_prefix_cache_counts_all_used_blocks() {
        let text = METRICS.replace("mini_engine_radix_tree_blocks{worker_id=\"w1\"} 312\n", "");
        let snap = snapshot_from_metrics(&text, &mut CounterRate::default(), Instant::now());
        assert_eq!(snap.models[0].kv_cache_usage, Some(824.0 / 1024.0));
    }
}
