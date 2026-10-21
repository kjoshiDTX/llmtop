# llmtop

Watches local LLM runtimes (Ollama, the llama.cpp server, vLLM) and the host they run on, and puts their metrics into one shared table.

```
CPU  41.9%   RAM 14.4 GiB / 16.0 GiB (90%)

BACKEND                    MODEL                          SIZE  PLACEMENT    KV  RUN WAIT   TOK/S
llama.cpp localhost:8089   qwen2.5-0.5b                      -  -             -    1    0    79.8
ollama localhost:11434     qwen2.5:0.5b                1.4 GiB  GPU           -    -    -       -
```

## Usage

```sh
cargo run -- --watch 2                                  # Ollama only, refresh every 2 s
cargo run -- --vllm gpu1:8000 --vllm gpu2:8000 -w 2     # several vLLM workers
cargo run -- --llamacpp localhost:8080 --no-ollama      # llama.cpp only, print once
```

Ollama is polled at `$OLLAMA_HOST` or `localhost:11434` unless `--no-ollama` is given. Tokens per second needs two polls, so it only appears in `--watch` mode.

## What each backend reports

| Column | Ollama (`/api/ps`) | vLLM (`/metrics`) | llama.cpp (`/metrics`, `/v1/models`) |
|---|---|---|---|
| SIZE, PLACEMENT | `size`, `size_vram` | – | – |
| KV | – | `kv_cache_usage_perc` (fullest engine) | only on older builds (`kv_cache_usage_ratio`) |
| RUN / WAIT | – | `num_requests_running` / `_waiting`, summed over engines | `requests_processing` / `requests_deferred` |
| TOK/S | – (see below) | rate of `generation_tokens_total` | rate of `n_decode_total` × `n_busy_slots_per_decode` |

- **Ollama throughput:** Ollama only reports token counts in the responses to your own requests, and has no endpoint for other clients' throughput. Getting it would take a proxy in front of Ollama or a periodic probe request.
- **llama.cpp throughput:** `tokens_predicted_total` only moves when a request finishes, so llmtop estimates from decode steps instead. The estimate is exact for one slot generating text.
- **llama.cpp needs `--metrics`**, or `/metrics` returns 501.

## Layout

| File | Role |
|---|---|
| `src/snapshot.rs` | The shared schema: `Snapshot`, `ModelStats`, `Placement` |
| `src/backend/mod.rs` | The `Backend` trait, counter-to-rate helper |
| `src/backend/{ollama,vllm,llamacpp}.rs` | One adapter per runtime |
| `src/prom.rs` | Prometheus text-format parser (zero-copy where possible) |
| `src/state.rs` | `Arc<Shared>` with an `RwLock<State>`; one tokio task per backend |
| `src/table.rs` | Text rendering |
| `fixtures/` | llama.cpp responses recorded from llama-server 0.6.0; a hand-written vLLM sample |

## Roadmap

| # | Build | Status |
|---|---|---|
| 1 | CLI that polls Ollama and sysinfo and prints a table | done |
| 2 | `Backend` trait with Ollama, vLLM and llama.cpp adapters, polled concurrently | done |
| 3 | Enum state machine that detects GPU→CPU fallback, KV cache filling, model swaps; replay tests | |
| 4 | axum `/metrics` exporter and a Grafana dashboard | |
| 5 | ratatui dashboard and cross-compiled releases from GitHub Actions | |
