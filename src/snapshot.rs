//! The shared schema every backend maps into. Fields are `Option` because no
//! runtime reports everything: Ollama knows where weights live but not
//! throughput, vLLM knows KV-cache pressure but not model size.

#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    pub models: Vec<ModelStats>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelStats {
    pub name: String,
    pub size_bytes: Option<u64>,
    pub placement: Option<Placement>,
    /// 0.0 to 1.0. With several engines, the fullest one.
    pub kv_cache_usage: Option<f64>,
    pub requests_running: Option<u64>,
    pub requests_waiting: Option<u64>,
    /// Generated tokens per second since the previous poll.
    pub gen_tokens_per_sec: Option<f64>,
}

impl ModelStats {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), ..Default::default() }
    }
}

/// Where a model's weights live. This is the signal behind the "fell back to
/// CPU" alert in a later milestone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Placement {
    Gpu,
    Cpu,
    /// Split between GPU and CPU; the value is the share on the GPU, 0.0 to 1.0.
    Split(f64),
}
