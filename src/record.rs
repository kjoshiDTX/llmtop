//! `--record FILE`: saves every raw backend response as one JSON line, so a
//! session can be replayed later through the same parsing and detection code.
//!
//! ```text
//! {"backend":"ollama localhost:11434","t_ms":1003,"raw":"{\"models\":[...]}"}
//! {"backend":"vllm gpu1:8000","t_ms":1004,"error":"could not reach ..."}
//! ```

use crate::backend::BackendId;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

/// One line of a recording. Generic over `R`, the backend's `Raw` type: the
/// recorder writes `Line<&R>` (borrowing, no copy), replays read `Line<R>`.
#[derive(Debug, Serialize, Deserialize)]
pub struct Line<R> {
    pub backend: String,
    /// Milliseconds since recording started.
    pub t_ms: u64,
    // A missing `Option` field reads as `None` without `#[serde(default)]`,
    // which here would wrongly require `R: Default`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<R>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub struct Recorder {
    /// Every poller writes to the same file, so access goes through a lock.
    file: Mutex<File>,
    start: Instant,
}

impl Recorder {
    pub fn create(path: &Path) -> Result<Self> {
        let file = File::create(path).with_context(|| format!("can't create {}", path.display()))?;
        Ok(Self { file: Mutex::new(file), start: Instant::now() })
    }

    pub fn write<R: Serialize>(&self, backend: &BackendId, at: Instant, fetched: &Result<R>) -> Result<()> {
        let line = Line {
            backend: backend.to_string(),
            t_ms: at.saturating_duration_since(self.start).as_millis() as u64,
            raw: fetched.as_ref().ok(),
            error: fetched.as_ref().err().map(|e| format!("{e:#}")),
        };
        let mut json = serde_json::to_string(&line)?;
        json.push('\n');
        // One `write_all` per line, so lines from different pollers never interleave.
        self.file.lock().expect("recorder lock poisoned").write_all(json.as_bytes())?;
        Ok(())
    }
}
