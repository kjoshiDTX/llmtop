//! Host CPU and RAM via the sysinfo crate.

use std::time::Instant;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System, MINIMUM_CPU_UPDATE_INTERVAL};

#[derive(Debug, Clone, Copy)]
pub struct SystemStats {
    pub cpu_percent: f32,
    pub mem_used: u64,
    pub mem_total: u64,
}

/// Owns a `System` across polls. CPU usage is a delta between two refreshes,
/// so the sampler has to keep state rather than build a fresh `System` each time.
pub struct SystemSampler {
    sys: System,
    last_refresh: Instant,
}

impl SystemSampler {
    pub fn new() -> Self {
        let sys = System::new_with_specifics(
            RefreshKind::nothing()
                .with_cpu(CpuRefreshKind::nothing().with_cpu_usage())
                .with_memory(MemoryRefreshKind::nothing().with_ram()),
        );
        Self { sys, last_refresh: Instant::now() }
    }

    /// Takes `&mut self` because refreshing mutates the inner `System`.
    pub async fn sample(&mut self) -> SystemStats {
        // The first CPU reading is meaningless until enough time has passed
        // since the previous refresh, so wait out the remainder if needed.
        let elapsed = self.last_refresh.elapsed();
        if elapsed < MINIMUM_CPU_UPDATE_INTERVAL {
            tokio::time::sleep(MINIMUM_CPU_UPDATE_INTERVAL - elapsed).await;
        }
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        self.last_refresh = Instant::now();

        SystemStats {
            cpu_percent: self.sys.global_cpu_usage(),
            mem_used: self.sys.used_memory(),
            mem_total: self.sys.total_memory(),
        }
    }
}
