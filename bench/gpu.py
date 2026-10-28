"""Background nvidia-smi sampler.

Under WSL2, nvidia-smi cannot attribute memory to processes ("No running processes found"),
so every memory number here is device-wide memory.used, which also includes what the
Windows desktop holds. study.py records an idle baseline before each server starts.
"""

import statistics
import subprocess
import threading
import time

FIELDS = ["memory.used", "memory.total", "utilization.gpu", "clocks.sm", "clocks.mem",
          "temperature.gpu", "power.draw", "clocks_event_reasons.active", "pstate"]


def _num(s: str) -> float | None:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


class GpuSampler:
    def __init__(self, period_ms: int = 100):
        self.period_ms = period_ms
        self.samples: list[dict] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self.t0 = 0.0

    def start(self) -> "GpuSampler":
        self.t0 = time.perf_counter()
        self._proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits",
             f"-lms={self.period_ms}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            vals = [v.strip() for v in line.split(",")]
            if len(vals) != len(FIELDS):
                continue
            s = {"t": round(time.perf_counter() - self.t0, 3)}
            for k, v in zip(FIELDS, vals):
                s[k] = v if k in ("clocks_event_reasons.active", "pstate") else _num(v)
            self.samples.append(s)

    def stop(self) -> list[dict]:
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        if self._thread:
            self._thread.join(timeout=5)
        return self.samples


def sample_for(seconds: float, period_ms: int = 100) -> list[dict]:
    s = GpuSampler(period_ms).start()
    time.sleep(seconds)
    return s.stop()


def summarize(samples: list[dict]) -> dict:
    def col(k):
        return [s[k] for s in samples if s.get(k) is not None]
    mem, util, sm = col("memory.used"), col("utilization.gpu"), col("clocks.sm")
    busy_sm = [s["clocks.sm"] for s in samples
               if s.get("clocks.sm") is not None and (s.get("utilization.gpu") or 0) >= 50]
    reasons = sorted({s["clocks_event_reasons.active"] for s in samples
                      if s.get("clocks_event_reasons.active")})
    return {
        "n_samples": len(samples),
        "mem_used_mib_max": max(mem) if mem else None,
        "mem_used_mib_min": min(mem) if mem else None,
        "mem_total_mib": samples[0].get("memory.total") if samples else None,
        "util_pct_mean": statistics.fmean(util) if util else None,
        "sm_clock_mhz_mean_busy": statistics.fmean(busy_sm) if busy_sm else None,
        "sm_clock_mhz_min_busy": min(busy_sm) if busy_sm else None,
        "temp_c_max": max(col("temperature.gpu"), default=None),
        "power_w_mean": statistics.fmean(col("power.draw")) if col("power.draw") else None,
        "clock_event_reasons_seen": reasons,
    }
