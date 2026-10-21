//! Replays `--record` files through the real `parse` and `Detector` code.
//! The recordings in `fixtures/replay/` were captured from live servers.

use crate::backend::Backend;
use crate::detect::{Detector, Event, Thresholds};
use crate::record::Line;
use anyhow::anyhow;
use std::time::{Duration, Instant};

/// Feeds every line recorded for `backend` into it and a fresh detector,
/// using the recorded timestamps, and returns the events in order.
fn replay<B: Backend>(mut backend: B, recording: &str) -> Vec<Event> {
    let mut detector = Detector::new(Thresholds::default());
    let t0 = Instant::now();
    let mut events = Vec::new();
    for (n, text) in recording.lines().enumerate() {
        // `B::Raw` names the adapter's own raw type, so serde knows what to decode.
        let line: Line<B::Raw> =
            serde_json::from_str(text).unwrap_or_else(|e| panic!("line {}: {e}", n + 1));
        if line.backend != backend.name() {
            continue;
        }
        let now = t0 + Duration::from_millis(line.t_ms);
        let result = match (line.raw, line.error) {
            (Some(raw), _) => backend.parse(&raw, now),
            (None, error) => Err(anyhow!(error.unwrap_or_default())),
        };
        events.extend(detector.observe(&result, now));
    }
    events
}

use crate::backend::{LlamaCpp, Ollama};

fn ollama() -> Ollama {
    Ollama::new("localhost:11434", reqwest::Client::new())
}

/// Ollama 0.7.0 on Apple Silicon, polled every second while qwen2.5:0.5b was
/// reloaded with `num_gpu` 0 (all CPU), then 12 (87% GPU), then 99 (all GPU).
/// During the first reload, one poll caught the model list empty.
#[test]
fn ollama_cpu_fallback() {
    let events = replay(ollama(), include_str!("../fixtures/replay/ollama_cpu_fallback.jsonl"));
    let model = || "qwen2.5:0.5b".to_string();
    assert_eq!(
        events,
        vec![
            // Gone for one poll, then back under the same name.
            Event::ModelReloaded { model: model() },
            Event::FellBackToCpu { model: model(), gpu_share: 0.0 },
            // Moving from all-CPU to 87% GPU is still partly on CPU: no new event.
            Event::BackOnGpu { model: model() },
        ]
    );
}

/// Ollama 0.7.0 switching between qwen2.5:0.5b and a tag of the same weights
/// with a different context size, which forces an unload and a load.
#[test]
fn ollama_model_swaps() {
    let events = replay(ollama(), include_str!("../fixtures/replay/ollama_swap.jsonl"));
    let qwen = || "qwen2.5:0.5b".to_string();
    let alt = || "llmtop-swap-test:latest".to_string();
    assert_eq!(
        events,
        vec![
            Event::ModelSwapped { from: qwen(), to: alt() },
            Event::ModelSwapped { from: alt(), to: qwen() },
        ]
    );
}

/// llama-server 0.6.0 with one 2048-token slot, while a single request
/// generated until the context was full (40 prompt + 2008 generated tokens).
#[test]
fn llamacpp_kv_cache_fills_and_clears() {
    let backend = LlamaCpp::new("localhost:8089", reqwest::Client::new());
    let events = replay(backend, include_str!("../fixtures/replay/llamacpp_kv_fill.jsonl"));
    assert_eq!(
        events,
        vec![
            // 1890 of 2048 tokens, the first poll at or above 90%.
            Event::KvCacheHigh { model: "qwen2.5-0.5b".into(), usage: 1890.0 / 2048.0 },
            // The request ended and the slot went idle.
            Event::KvCacheRecovered { model: "qwen2.5-0.5b".into(), usage: 0.0 },
        ]
    );
}
