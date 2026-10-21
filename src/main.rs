mod backend;
mod detect;
mod prom;
mod record;
#[cfg(test)]
mod replay;
mod snapshot;
mod state;
mod system;
mod table;

use anyhow::Result;
use backend::{LlamaCpp, Ollama, Vllm};
use clap::Parser;
use record::Recorder;
use state::{spawn_poller, spawn_system_sampler, PollConfig, Shared};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// Watch local LLM runtimes and the host they run on.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Ollama server URL.
    #[arg(long, env = "OLLAMA_HOST", default_value = "http://localhost:11434")]
    ollama: String,

    /// Don't poll Ollama.
    #[arg(long)]
    no_ollama: bool,

    /// vLLM server URL. Repeat for several workers.
    #[arg(long, value_name = "URL")]
    vllm: Vec<String>,

    /// llama.cpp server URL (started with --metrics). Repeatable.
    #[arg(long, value_name = "URL")]
    llamacpp: Vec<String>,

    /// Refresh every N seconds instead of printing once.
    #[arg(short, long, value_name = "SECS")]
    watch: Option<u64>,

    /// Append every raw backend response to FILE as JSON lines, for replay tests.
    #[arg(long, value_name = "FILE")]
    record: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let every = args.watch.map(|secs| Duration::from_secs(secs.max(1)));
    let shared = Arc::new(Shared::default());
    // Subscribe before any task starts, so no early update is missed.
    let mut changed = shared.changed.subscribe();
    let http = backend::http_client()?;
    let config = PollConfig {
        every,
        thresholds: detect::Thresholds::default(),
        recorder: args.record.as_deref().map(Recorder::create).transpose()?.map(Arc::new),
    };

    // `Arc::clone` copies the pointer, not the state: every task shares one `Shared`.
    let mut tasks = vec![spawn_system_sampler(Arc::clone(&shared), every)];
    if !args.no_ollama {
        let ollama = Ollama::new(&args.ollama, http.clone());
        tasks.push(spawn_poller(ollama, Arc::clone(&shared), config.clone()));
    }
    for url in &args.vllm {
        tasks.push(spawn_poller(Vllm::new(url, http.clone()), Arc::clone(&shared), config.clone()));
    }
    for url in &args.llamacpp {
        let llamacpp = LlamaCpp::new(url, http.clone());
        tasks.push(spawn_poller(llamacpp, Arc::clone(&shared), config.clone()));
    }

    if every.is_none() {
        for task in tasks {
            task.await?;
        }
        print!("{}", render(&shared));
        return Ok(());
    }

    loop {
        changed.changed().await?;
        // Backends polled on the same tick finish within milliseconds of each
        // other; a short pause lets them land in one frame instead of several.
        tokio::time::sleep(Duration::from_millis(100)).await;
        changed.borrow_and_update();
        // Clear the screen and move the cursor home before each frame.
        print!("\x1b[2J\x1b[H{}", render(&shared));
    }
}

/// The read guard lives only for this call, so pollers are never blocked
/// while the frame is printed.
fn render(shared: &Shared) -> String {
    table::render(&shared.state.read().expect("state lock poisoned"))
}
