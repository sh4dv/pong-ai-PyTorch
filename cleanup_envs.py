"""
Utility to clean up stray Pong training processes and GPU caches.
Run after an interrupted training run to terminate orphaned AsyncVectorEnv workers
and free GPU/MPS memory.

Usage examples:
  python cleanup_envs.py            # show what would be cleaned
  python cleanup_envs.py --kill     # actually terminate matched processes
  python cleanup_envs.py --kill --signal KILL  # force kill if TERM is not enough
"""

import argparse
import gc
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# Optional psutil support for richer process metadata
try:
    import psutil  # type: ignore

    HAS_PSUTIL = True
except ImportError:  # pragma: no cover - optional dependency
    psutil = None
    HAS_PSUTIL = False

REPO_ROOT = Path(__file__).resolve().parent
# Keywords that indicate the process belongs to this project
TARGET_KEYWORDS = [
    "train.py",
    "debug_agent.py",
    "benchmark_parallel.py",
    "benchmark_envs.py",
    "benchmark_gpu.py",
    "pong_env.py",
    "pong_game.py",
    "gymnasium.vector.async_vector_env",
]
PYTHON_NAMES = {"python", "python3", "python3.11", "python3.10"}


def _within_repo(cwd: Optional[str]) -> bool:
    if not cwd:
        return False
    try:
        path = Path(cwd).resolve()
    except OSError:
        return False
    return path == REPO_ROOT or REPO_ROOT in path.parents


def _format_cmd(cmdline: List[str]) -> str:
    if not cmdline:
        return "<unknown>"
    return " ".join(cmdline)


def _find_candidates_psutil():
    candidates = []
    for proc in psutil.process_iter(attrs=["pid", "ppid", "name", "cmdline", "cwd"]):
        if proc.info["pid"] == os.getpid():
            continue
        name = proc.info.get("name") or ""
        cmdline = proc.info.get("cmdline") or []
        cmd_str = _format_cmd(cmdline)
        if not cmdline:
            continue
        if name not in PYTHON_NAMES and not any(part in name for part in PYTHON_NAMES):
            continue
        if not any(key in cmd_str for key in TARGET_KEYWORDS) and REPO_ROOT.as_posix() not in cmd_str:
            continue
        if not _within_repo(proc.info.get("cwd")) and REPO_ROOT.as_posix() not in cmd_str:
            continue
        candidates.append(proc)
    return candidates


def _find_candidates_ps():
    candidates = []
    try:
        output = subprocess.check_output(["ps", "-axo", "pid,ppid,command"], text=True)
    except subprocess.SubprocessError:
        return candidates
    for line in output.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid_str, _, cmd = parts
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        if not any(key in cmd for key in TARGET_KEYWORDS) and REPO_ROOT.as_posix() not in cmd:
            continue
        if not any(name in cmd for name in PYTHON_NAMES):
            continue
        candidates.append((pid, cmd))
    return candidates


def find_candidates():
    if HAS_PSUTIL:
        return _find_candidates_psutil()
    return _find_candidates_ps()


def terminate_process(proc, sig: signal.Signals, timeout: float) -> bool:
    if HAS_PSUTIL:
        try:
            proc.send_signal(sig)
            proc.wait(timeout=timeout)
            return True
        except psutil.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=2)
                return True
            except Exception:
                return False
        except Exception:
            return False
    else:
        pid, _ = proc
        try:
            os.kill(pid, sig)
            time.sleep(timeout)
            return True
        except ProcessLookupError:
            return True
        except Exception:
            return False


def clear_torch_cache():
    try:
        import torch  # type: ignore
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache.")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            print("Cleared MPS cache.")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: cache cleanup failed: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Cleanup stray Pong training processes and memory caches.")
    parser.add_argument("--kill", action="store_true", help="Terminate matched processes instead of listing.")
    parser.add_argument("--signal", default="TERM", choices=["TERM", "KILL"], help="Signal used with --kill.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Seconds to wait after sending the signal.")
    parser.add_argument("--no-cache", action="store_true", help="Skip clearing torch GPU/MPS caches.")
    args = parser.parse_args()

    print(f"Repo root detected at: {REPO_ROOT}")
    print("Scanning for stray env/training processes...")

    candidates = find_candidates()
    if not candidates:
        print("No matching processes found. Nothing to do.")
    else:
        if HAS_PSUTIL:
            for proc in candidates:
                cmd_str = _format_cmd(proc.info.get("cmdline") or [])
                cwd = proc.info.get("cwd") or "<unknown>"
                print(f"PID {proc.pid} (cwd={cwd}) -> {cmd_str}")
        else:
            for pid, cmd in candidates:
                print(f"PID {pid} -> {cmd}")

        if args.kill:
            sig = signal.SIGTERM if args.signal == "TERM" else signal.SIGKILL
            print(f"\nSending {sig.name} to {len(candidates)} process(es)...")
            for proc in candidates:
                success = terminate_process(proc, sig=sig, timeout=args.timeout)
                if HAS_PSUTIL:
                    print(f"  PID {proc.pid}: {'terminated' if success else 'failed'}")
                else:
                    pid, _ = proc
                    print(f"  PID {pid}: {'terminated' if success else 'failed'}")
        else:
            print("\nDry run only. Re-run with --kill to terminate them.")

    # Memory cleanup happens even when nothing was found
    gc.collect()
    if not args.no_cache:
        clear_torch_cache()

    print("Cleanup done.")


if __name__ == "__main__":
    sys.exit(main())
