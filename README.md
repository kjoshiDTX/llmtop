# llmtop

Watches local LLM runtimes (Ollama, the llama.cpp server, vLLM, and llm-inference-scheduler's mini-engine workers) and the host they run on. It puts their metrics into one shared table, raises plain-language alerts when something changes, and can serve everything to Prometheus and Grafana.

Example output, assembled from several runs:

```
CPU  62.1%   RAM 14.0 GiB / 16.0 GiB (88%)

BACKEND                    MODEL                          SIZE  PLACEMENT    KV  RUN WAIT   TOK/S
llama.cpp localhost:8089   qwen2.5-0.5b                      -  -             -    1    0    79.8
mini-engine localhost:8001 w1                                -  -           79%    7    1    35.1
ollama localhost:11434     qwen2.5:0.5b                1.4 GiB  GPU           -    -    -       -

ALERTS
17:39:52 ! ollama localhost:11434: qwen2.5:0.5b is running entirely on CPU; generation will be much slower
17:53:40 ! mini-engine localhost:8001: w1 KV cache is 92% full; new requests may queue or be preempted
17:53:46   mini-engine localhost:8001: w1 KV cache is back down to 79%
```

## Install

You need **git** and **Rust 1.95 or newer** (required by the `sysinfo` dependency). If you don't have Rust, install it with [rustup](https://rustup.rs); if you already do, `rustup update` gets the latest stable:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then clone and install:

```sh
git clone https://github.com/kjoshiDTX/llmtop.git
cd llmtop
cargo install --path .
```

`cargo install` builds a release binary and puts `llmtop` in `~/.cargo/bin`, which rustup adds to your `PATH`. Check it worked:

```sh
llmtop --help
```

To build without installing, run `cargo build --release` and use `./target/release/llmtop`. During development, `cargo run -- <flags>` builds and runs in one step.

The repository is private, so cloning needs a GitHub account with access to it. With the GitHub CLI signed in, `gh repo clone kjoshiDTX/llmtop` also works.

## Use

llmtop polls whichever runtimes you point it at. With no flags, it polls Ollama once and prints a table. Add `-w SECS` to refresh continuously, which also turns on tokens per second (it needs two polls) and alerts.

| Runtime | Start it so llmtop can read it | Point llmtop at it |
|---|---|---|
| Ollama | `ollama serve` (the desktop app does this) | on by default; `--ollama URL` or `$OLLAMA_HOST` to change, `--no-ollama` to skip |
| llama.cpp | `llama-server -m model.gguf --port 8080 --metrics` | `--llamacpp localhost:8080` |
| vLLM | `vllm serve <model>` (metrics are on its API port) | `--vllm localhost:8000` |
| mini-engine | `WORKER_ID=w1 .venv/bin/python -m uvicorn server.worker:app --port 8001`, from llm-inference-scheduler | `--mini-engine localhost:8001` |

Every flag except `--ollama` can be repeated to watch several servers. For example:

```sh
llmtop -w 2                                                          # Ollama, refresh every 2 s
llmtop -w 2 --llamacpp localhost:8080                                # Ollama and llama.cpp
llmtop -w 2 --no-ollama --mini-engine localhost:8001 --mini-engine localhost:8002   # the scheduler's two workers
llmtop -w 2 --vllm gpu1:8000 --vllm gpu2:8000                        # vLLM workers on other machines
llmtop -w 1 --record session.jsonl                                   # also save raw responses for replay tests
```

Press Ctrl-C to stop.

## Prometheus and Grafana

`--listen ADDR` serves every metric on `http://ADDR/metrics`. Without `-w`, llmtop runs headless and polls every 5 seconds; with `-w`, it shows the table too.

The `deploy/` folder has a Docker Compose setup with Prometheus and a provisioned Grafana dashboard. You need [Docker](https://docs.docker.com/get-docker/) running.

1. Start llmtop on the host, pointed at your workers:

   ```sh
   llmtop --no-ollama --mini-engine localhost:8001 --mini-engine localhost:8002 --listen 0.0.0.0:9898
   ```

   Prometheus runs in a container and reaches the host through `host.docker.internal`, so llmtop has to accept connections from outside loopback. `0.0.0.0` does that, and also exposes port 9898 to your network. Use a firewall if that matters on your network.

2. From the repository root, start Prometheus and Grafana:

   ```sh
   docker compose -f deploy/docker-compose.yml up -d
   ```

3. Open http://localhost:3000. The llmtop dashboard is the home page, with no login needed to view it. To edit it, sign in as `admin` / `admin`; Grafana asks you to change that on first sign-in. If panels stay empty, open http://localhost:9090/targets and check that the `llmtop` target is up.

The dashboard shows throughput, KV-cache usage with the 80% and 90% alert lines, running and queued requests, backend status over time, alert counts by type, GPU placement (Ollama), and host CPU and memory. Warnings also appear as annotations on every time-series panel. Use the **Backend kind** and **Endpoint** menus at the top to filter.

Stop it with `docker compose -f deploy/docker-compose.yml down`. Prometheus keeps its data in a Docker volume between runs; add `-v` to delete it too.

### Exported metrics

Every backend metric has `kind` and `endpoint` labels (e.g. `mini-engine`, `localhost:8001`); per-model metrics add `model`. The label is `endpoint` rather than `instance` because Prometheus sets `instance` itself, to the address of llmtop.

| Metric | Type | Meaning |
|---|---|---|
| `llmtop_backend_up` | gauge | 1 if the last poll succeeded |
| `llmtop_model_loaded` | gauge | 1 for each model a backend reports |
| `llmtop_model_size_bytes` | gauge | Memory the model occupies (Ollama) |
| `llmtop_model_gpu_ratio` | gauge | Share of weights in GPU memory (Ollama) |
| `llmtop_kv_cache_usage_ratio` | gauge | Share of the KV cache in use |
| `llmtop_requests_running`, `llmtop_requests_waiting` | gauge | Requests generating and queued |
| `llmtop_generation_tokens_per_second` | gauge | Generated tokens per second since the previous poll |
| `llmtop_alerts_total` | counter | Alerts raised, labelled by `alert` (e.g. `kv_cache_high`); starts at 0 |
| `llmtop_host_cpu_usage_ratio`, `llmtop_host_memory_used_bytes`, `llmtop_host_memory_total_bytes` | gauge | The machine llmtop runs on |

## What each backend reports

| Column | Ollama (`/api/ps`) | vLLM (`/metrics`) | llama.cpp (`/metrics`, `/slots`, `/v1/models`) | mini-engine (`/metrics`) |
|---|---|---|---|---|
| MODEL | model name | `model_name` label | `/v1/models` | `worker_id` label (the worker doesn't report its model) |
| SIZE, PLACEMENT | `size`, `size_vram` | – | – | – |
| KV | – | `kv_cache_usage_perc` (fullest engine) | tokens held by busy slots ÷ total slot context | blocks that are neither free nor in the prefix cache ÷ total blocks |
| RUN / WAIT | – | `num_requests_running` / `_waiting`, summed over engines | `requests_processing` / `requests_deferred` | `num_running` / `num_waiting` |
| TOK/S | – (see below) | rate of `generation_tokens_total` | rate of `n_decode_total` × `n_busy_slots_per_decode` | rate of `tokens_generated` |

- **Ollama throughput:** Ollama only reports token counts in the responses to your own requests, and has no endpoint for other clients' throughput. Getting it would take a proxy in front of Ollama or a periodic probe request.
- **llama.cpp throughput:** `tokens_predicted_total` only moves when a request finishes, so llmtop estimates from decode steps instead. The estimate is exact for one slot generating text.
- **llama.cpp needs `--metrics`**, or `/metrics` returns 501. Current releases don't export KV-cache usage, so it comes from `/slots`. Idle slots count as empty even when they keep a cached prompt, because that cache can be reused.
- **mini-engine KV:** prefix-cache blocks are evictable, so they count as free, as in vLLM. Cached prefixes pinned by running requests are counted as free too, so the figure can read slightly low.

## Alerts

Each backend's polling task runs a `Detector` (`src/detect.rs`), which turns snapshots into events on transitions only:

| Event | When |
|---|---|
| Running partly/entirely on CPU, back on GPU | Ollama's `size_vram` drops below `size`, or recovers. A model first seen on CPU alerts too |
| KV cache high, back down | Usage reaches 90%; clears below 80%, so a cache hovering near 90% doesn't flap |
| Swapped out for X | A model disappears and another appears within 30 s |
| Reloaded | The same model disappears and comes back within 30 s |
| Loaded, unloaded | A model appears with nothing departed, or is gone for more than 30 s |
| Unreachable, reachable again | The poll fails, or succeeds after failing |

Models already loaded when llmtop starts are the baseline, not "loaded" events.

## Development

```sh
cargo test      # unit tests and replays of recorded sessions
cargo clippy --all-targets
```

The replay tests feed files from `fixtures/replay/` through the real parsers and detector. To add one, record a session with `--record FILE` while you make something happen, copy the file into `fixtures/replay/`, and add a test in `src/replay.rs` that lists the events you expect.

| File | Role |
|---|---|
| `src/snapshot.rs` | The shared schema: `Snapshot`, `ModelStats`, `Placement` |
| `src/backend/mod.rs` | The `Backend` trait, `BackendId`, counter-to-rate helper |
| `src/backend/{ollama,vllm,llamacpp,mini_engine}.rs` | One adapter per runtime |
| `src/prom.rs` | Prometheus text-format parser (zero-copy where possible) |
| `src/state.rs` | `Arc<Shared>` with an `RwLock<State>`; one tokio task per backend |
| `src/detect.rs` | Enum state machines that turn snapshots into alerts |
| `src/exporter.rs` | axum server and Prometheus text encoder for `--listen` |
| `src/record.rs` | `--record`: raw responses as JSON lines |
| `src/replay.rs` | Tests that replay recordings through the real parsers and detector |
| `src/table.rs` | Text rendering |
| `deploy/` | Docker Compose for Prometheus and Grafana, with the provisioned dashboard |
| `fixtures/` | llama.cpp responses recorded from llama-server 0.6.0; a hand-written vLLM sample |
| `fixtures/replay/` | Recorded sessions: Ollama CPU fallback, Ollama model swaps, llama.cpp KV cache filling |

## Roadmap

| # | Build | Status |
|---|---|---|
| 1 | CLI that polls Ollama and sysinfo and prints a table | done |
| 2 | `Backend` trait with Ollama, vLLM and llama.cpp adapters, polled concurrently | done |
| 3 | Enum state machine that detects GPU→CPU fallback, KV cache filling, model swaps; replay tests | done |
| 4 | axum `/metrics` exporter and a Grafana dashboard | done |
| 5 | ratatui dashboard and cross-compiled releases from GitHub Actions | |
