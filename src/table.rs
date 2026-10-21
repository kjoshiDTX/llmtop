//! Renders the shared state as plain text. Returns a `String` instead of
//! printing so tests can check the output.

use crate::detect::Severity;
use crate::snapshot::{ModelStats, Placement};
use crate::state::{BackendStatus, State};
use std::fmt::Write;

const HEADER: [&str; 8] = ["BACKEND", "MODEL", "SIZE", "PLACEMENT", "KV", "RUN", "WAIT", "TOK/S"];

/// How many of the most recent alerts to show under the table.
const ALERTS_SHOWN: usize = 8;

pub fn render(state: &State) -> String {
    let mut out = String::new();

    // `writeln!` into a String can't fail, so the results are safe to ignore.
    match &state.system {
        Some(sys) => {
            let _ = writeln!(
                out,
                "CPU {:>5.1}%   RAM {} / {} ({:.0}%)\n",
                sys.cpu_percent,
                human_bytes(sys.mem_used),
                human_bytes(sys.mem_total),
                percent(sys.mem_used, sys.mem_total),
            );
        }
        None => out.push_str("CPU/RAM: sampling…\n\n"),
    }

    if state.backends.is_empty() {
        out.push_str("No backends configured.\n");
        return out;
    }

    let _ = writeln!(out, "{}", row(HEADER.map(String::from)));
    for (id, status) in &state.backends {
        let name = truncate(&id.to_string(), 26);
        match status {
            BackendStatus::Pending => {
                let _ = writeln!(out, "{name:<26} polling…");
            }
            BackendStatus::Down(err) => {
                let _ = writeln!(out, "{name:<26} down: {err}");
            }
            BackendStatus::Up(snap) if snap.models.is_empty() => {
                let _ = writeln!(out, "{name:<26} no models loaded");
            }
            BackendStatus::Up(snap) => {
                for m in &snap.models {
                    let _ = writeln!(out, "{}", row(model_cells(&name, m)));
                }
            }
        }
    }

    if !state.alerts.is_empty() {
        out.push_str("\nALERTS\n");
        let skip = state.alerts.len().saturating_sub(ALERTS_SHOWN);
        for alert in state.alerts.iter().skip(skip) {
            let mark = match alert.event.severity() {
                Severity::Warning => '!',
                Severity::Info => ' ',
            };
            let _ = writeln!(
                out,
                "{} {mark} {}: {}",
                alert.at.format("%H:%M:%S"),
                alert.backend,
                alert.event
            );
        }
    }
    out
}

fn row(c: [String; 8]) -> String {
    format!(
        "{:<26} {:<26} {:>8}  {:<10} {:>4} {:>4} {:>4} {:>7}",
        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]
    )
}

fn model_cells(backend: &str, m: &ModelStats) -> [String; 8] {
    [
        backend.to_string(),
        truncate(&m.name, 26),
        opt(m.size_bytes, human_bytes),
        opt(m.placement, placement_label),
        opt(m.kv_cache_usage, |kv| format!("{:.0}%", kv * 100.0)),
        opt(m.requests_running, |n| n.to_string()),
        opt(m.requests_waiting, |n| n.to_string()),
        opt(m.gen_tokens_per_sec, |t| format!("{t:.1}")),
    ]
}

/// Formats a value the backend reported, or `-` for one it doesn't expose.
fn opt<T>(value: Option<T>, fmt: impl FnOnce(T) -> String) -> String {
    value.map_or_else(|| "-".to_string(), fmt)
}

fn placement_label(p: Placement) -> String {
    match p {
        Placement::Gpu => "GPU".to_string(),
        Placement::Cpu => "CPU".to_string(),
        Placement::Split(gpu) => format!("{:.0}% GPU", gpu * 100.0),
    }
}

fn percent(part: u64, whole: u64) -> f64 {
    if whole == 0 { 0.0 } else { part as f64 / whole as f64 * 100.0 }
}

fn human_bytes(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.1} GiB", b / GIB)
    } else {
        format!("{:.0} MiB", b / MIB)
    }
}

/// Truncates on character boundaries; slicing a `&str` by byte index could
/// panic in the middle of a multi-byte character.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max - 1).collect();
        t.push('…');
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendId;
    use crate::snapshot::Snapshot;

    fn id(kind: &'static str, instance: &str) -> BackendId {
        BackendId { kind, instance: instance.into() }
    }

    #[test]
    fn formats_bytes() {
        assert_eq!(human_bytes(397 * 1024 * 1024), "397 MiB");
        assert_eq!(human_bytes(6 * 1024 * 1024 * 1024), "6.0 GiB");
    }

    #[test]
    fn labels_split_placement() {
        assert_eq!(placement_label(Placement::Split(0.75)), "75% GPU");
    }

    #[test]
    fn renders_each_backend_status() {
        let mut state = State::default();
        state.backends.insert(id("a", "pending"), BackendStatus::Pending);
        state.backends.insert(id("b", "down"), BackendStatus::Down("connection refused".into()));
        state.backends.insert(id("c", "empty"), BackendStatus::Up(Snapshot::default()));
        let model = ModelStats { kv_cache_usage: Some(0.82), ..ModelStats::new("qwen") };
        state.backends.insert(id("d", "up"), BackendStatus::Up(Snapshot { models: vec![model] }));

        let out = render(&state);
        assert!(out.contains("CPU/RAM: sampling…"));
        assert!(out.contains("polling…"));
        assert!(out.contains("down: connection refused"));
        assert!(out.contains("no models loaded"));
        let qwen = out.lines().find(|l| l.contains("qwen")).unwrap();
        assert!(qwen.contains("82%"));
        assert!(!out.contains("ALERTS"));
    }

    #[test]
    fn renders_alerts_newest_last_with_warning_mark() {
        use crate::detect::Event;
        use crate::state::Alert;

        let mut state = State::default();
        state.backends.insert(id("ollama", "x"), BackendStatus::Up(Snapshot::default()));
        for i in 0..10 {
            state.alerts.push_back(Alert {
                at: chrono::Local::now(),
                backend: id("ollama", "x"),
                event: Event::ModelLoaded { model: format!("m{i}") },
            });
        }
        state.alerts.push_back(Alert {
            at: chrono::Local::now(),
            backend: id("ollama", "x"),
            event: Event::FellBackToCpu { model: "q".into(), gpu_share: 0.0 },
        });

        let out = render(&state);
        let alerts: Vec<&str> = out.lines().skip_while(|l| *l != "ALERTS").skip(1).collect();
        assert_eq!(alerts.len(), ALERTS_SHOWN);
        assert!(alerts[0].ends_with("ollama x: m3 loaded"));
        assert!(alerts[7].contains(" ! ollama x: q is running entirely on CPU"));
    }
}
