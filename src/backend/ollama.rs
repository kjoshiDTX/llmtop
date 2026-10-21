//! Ollama, via `/api/ps`, which lists the models currently loaded into memory.
//!
//! Ollama has no endpoint for throughput: token counts only come back in the
//! responses to your own requests. So this adapter leaves tokens per second empty.

use super::{normalize_url, Backend};
use crate::snapshot::{ModelStats, Placement, Snapshot};
use anyhow::{Context, Result};
use serde::Deserialize;

/// Mirrors the JSON from `GET /api/ps`. Only the fields llmtop uses are
/// declared; serde ignores the rest, so new Ollama fields won't break parsing.
#[derive(Debug, Deserialize)]
struct PsResponse {
    models: Vec<LoadedModel>,
}

#[derive(Debug, Deserialize)]
struct LoadedModel {
    name: String,
    /// Total bytes the model occupies, across GPU and system memory.
    size: u64,
    /// Bytes that sit in GPU memory. Less than `size` means a CPU fallback.
    #[serde(default)]
    size_vram: u64,
}

impl LoadedModel {
    fn placement(&self) -> Placement {
        if self.size == 0 || self.size_vram >= self.size {
            Placement::Gpu
        } else if self.size_vram == 0 {
            Placement::Cpu
        } else {
            Placement::Split(self.size_vram as f64 / self.size as f64)
        }
    }

    /// Consumes the parsed model: `self` by value lets us move `name` into
    /// the result instead of cloning it.
    fn into_stats(self) -> ModelStats {
        ModelStats {
            placement: Some(self.placement()),
            size_bytes: Some(self.size),
            ..ModelStats::new(self.name)
        }
    }
}

pub struct Ollama {
    name: String,
    base_url: String,
    http: reqwest::Client,
}

impl Ollama {
    pub fn new(url: &str, http: reqwest::Client) -> Self {
        let (base_url, host) = normalize_url(url);
        Self { name: format!("ollama {host}"), base_url, http }
    }
}

impl Backend for Ollama {
    fn name(&self) -> &str {
        &self.name
    }

    async fn poll(&mut self) -> Result<Snapshot> {
        let url = format!("{}/api/ps", self.base_url);
        let resp: PsResponse = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("could not reach {url}"))?
            .error_for_status()?
            .json()
            .await
            .context("Ollama returned JSON in an unexpected shape")?;
        Ok(Snapshot { models: resp.models.into_iter().map(LoadedModel::into_stats).collect() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A response shape from Ollama's API docs, with one model partly on CPU.
    const PS_JSON: &str = r#"{
      "models": [
        {
          "name": "llama3.1:8b",
          "model": "llama3.1:8b",
          "size": 6000000000,
          "digest": "abc",
          "details": { "family": "llama", "parameter_size": "8.0B", "quantization_level": "Q4_K_M" },
          "expires_at": "2026-10-21T14:38:31.83753-07:00",
          "size_vram": 4500000000
        }
      ]
    }"#;

    #[test]
    fn maps_ps_response_to_stats() {
        let resp: PsResponse = serde_json::from_str(PS_JSON).unwrap();
        let stats: Vec<_> = resp.models.into_iter().map(LoadedModel::into_stats).collect();
        assert_eq!(stats[0].name, "llama3.1:8b");
        assert_eq!(stats[0].size_bytes, Some(6_000_000_000));
        assert_eq!(stats[0].placement, Some(Placement::Split(0.75)));
        assert_eq!(stats[0].gen_tokens_per_sec, None);
    }

    #[test]
    fn placement_edges() {
        let mut m: LoadedModel = serde_json::from_str(r#"{"name":"x","size":100}"#).unwrap();
        assert_eq!(m.placement(), Placement::Cpu);
        m.size_vram = 100;
        assert_eq!(m.placement(), Placement::Gpu);
    }
}
