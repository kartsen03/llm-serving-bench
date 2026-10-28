"""Records the hardware and software a result came from. Every run file embeds this."""

import importlib.metadata as md
import json
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path


def sh(cmd: list[str], timeout: float = 30) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or out.stderr).strip()
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def git_info(repo: Path) -> dict:
    g = ["git", "-c", "safe.directory=*", "-c", "core.filemode=false", "-C", str(repo)]
    dirty = sh(g + ["status", "--porcelain", "--untracked-files=no"])
    commit = sh(g + ["rev-parse", "--verify", "HEAD"])
    if not re.fullmatch(r"[0-9a-f]{40}", commit):  # no repo or no commits yet
        return {"commit": None, "dirty": True, "dirty_files": [commit[:200]]}
    return {"commit": commit, "dirty": bool(dirty), "dirty_files": dirty.splitlines()[:20]}


def gpu_info() -> dict:
    q = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,vbios_version,"
            "clocks.max.sm,clocks.max.mem", "--format=csv,noheader"]).split(", ")
    header = sh(["nvidia-smi"])
    cuda = re.search(r"CUDA Version:\s*(\S+)", header)
    smi = re.search(r"NVIDIA-SMI\s+(\S+)", header)
    keys = ["name", "driver_version", "memory_total", "vbios", "max_sm_clock", "max_mem_clock"]
    info = dict(zip(keys, q))
    info["cuda_driver_api_version"] = cuda.group(1) if cuda else None
    info["nvidia_smi_version"] = smi.group(1) if smi else None
    return info


def windows_info() -> dict:
    scheme = sh(["powercfg.exe", "/getactivescheme"])
    m = re.search(r"\(([^)]+)\)", scheme)
    return {"windows_ver": sh(["cmd.exe", "/c", "ver"]).replace("\r", "").strip(),
            "power_plan": m.group(1) if m else scheme}


def pkg(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def torch_cuda() -> str | None:
    out = sh([sys.executable, "-c", "import torch; print(torch.version.cuda)"], timeout=120)
    return out if re.fullmatch(r"[\d.]+", out) else None


def collect(repo: Path) -> dict:
    cpu = sh(["bash", "-c", "lscpu | grep 'Model name' | sed 's/.*: *//'"])
    mem = sh(["bash", "-c", "free -m | awk '/Mem:/ {print $2}'"])
    nvcc = sh(["/usr/local/cuda/bin/nvcc", "--version"]).splitlines()
    return {
        "git": git_info(repo),
        "gpu": gpu_info(),
        "cuda_toolkit_nvcc": nvcc[-2].strip() if len(nvcc) >= 2 else None,
        "host": {"cpu": cpu, "wsl_mem_mib": mem, "kernel": platform.release(),
                 "distro": sh(["bash", "-c", "grep PRETTY_NAME /etc/os-release | cut -d= -f2"]).strip('"'),
                 **windows_info()},
        "python": platform.python_version(),
        "packages": {p: pkg(p) for p in ["vllm", "torch", "transformers", "aiohttp",
                                         "flashinfer-python", "triton"]},
        "torch_cuda_runtime": torch_cuda(),
    }


def http_json(url: str, timeout: float = 5) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


if __name__ == "__main__":
    print(json.dumps(collect(Path(__file__).resolve().parents[1]), indent=2))
