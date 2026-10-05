"""Observe a running Linux viewer without touching its camera or event loop.

Usage: .venv/bin/python tools/watch_ui.py PID --output diagnostics/ui-watch.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def process_stat(text: str) -> dict:
    # The command name can contain spaces and closing parentheses.
    fields = text[text.rfind(")") + 2 :].split()
    return {
        "state": fields[0],
        "cpu_ticks": int(fields[11]) + int(fields[12]),
        "start_ticks": int(fields[19]),
        "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
    }


def read_optional(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=2, check=False)
        return (result.stdout or result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def usb_power_snapshot() -> dict:
    devices = {}
    for device in Path("/sys/bus/usb/devices").glob("*"):
        if read_optional(device / "idVendor") != "2bdf":
            continue
        path = device.resolve()
        while path != Path("/sys/devices") and path.is_relative_to("/sys/devices"):
            state = {
                key: read_optional(path / "power" / key)
                for key in (
                    "control",
                    "runtime_status",
                    "runtime_active_time",
                    "runtime_suspended_time",
                    "autosuspend_delay_ms",
                )
            }
            if any(value is not None for value in state.values()):
                devices[str(path)] = state
            path = path.parent
    return devices


def snapshot(pid: int) -> dict:
    root = Path(f"/proc/{pid}")
    stat = process_stat((root / "stat").read_text())
    threads = {}
    for task in (root / "task").iterdir():
        try:
            item = process_stat((task / "stat").read_text())
        except OSError:
            continue
        item["name"] = read_optional(task / "comm")
        item["wait_channel"] = read_optional(task / "wchan")
        item["scheduler"] = read_optional(task / "schedstat")
        threads[task.name] = item
    csv_files = {}
    for fd in (root / "fd").iterdir():
        try:
            target = fd.readlink()
            if target.suffix.lower() == ".csv":
                csv_files[str(target)] = target.stat().st_size
        except OSError:
            continue
    return {
        "event": "sample",
        "time": datetime.now(UTC).isoformat(),
        "monotonic": time.monotonic(),
        "process": stat,
        "threads": threads,
        "io": read_optional(root / "io"),
        "csv_bytes": csv_files,
        "usb_power": usb_power_snapshot(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=3600, help="Maximum seconds to monitor")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    previous = None
    main_idle_since = started
    next_display = started
    with args.output.open("a", buffering=1) as output:

        def write(record):
            output.write(json.dumps(record) + "\n")

        write(
            {
                "event": "start",
                "pid": args.pid,
                "time": datetime.now(UTC).isoformat(),
                "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
                "ptrace_scope": read_optional(Path("/proc/sys/kernel/yama/ptrace_scope")),
                "kernel_stack_access": read_optional(Path(f"/proc/{args.pid}/stack")) is not None,
                "notes": "CPU inactivity and CSV file sizes are clues, not frame counters. "
                "This monitor does not acquire the camera or change power settings.",
            }
        )
        while time.monotonic() - started < args.duration:
            try:
                record = snapshot(args.pid)
            except (OSError, IndexError, ValueError):
                write({"event": "process_exited", "time": datetime.now(UTC).isoformat()})
                return 0
            now = record["monotonic"]
            if record["process"]["state"] in ("Z", "X"):
                write({"event": "process_exited", "time": datetime.now(UTC).isoformat()})
                return 0
            if previous:
                if previous["process"]["start_ticks"] != record["process"]["start_ticks"]:
                    write({"event": "pid_reused"})
                    return 0
                elapsed = now - previous["monotonic"]
                ticks = os.sysconf("SC_CLK_TCK")
                for tid, thread in record["threads"].items():
                    before = previous["threads"].get(tid)
                    if before:
                        thread["cpu_percent"] = (
                            (thread["cpu_ticks"] - before["cpu_ticks"]) / ticks / elapsed * 100
                        )
                main_thread = record["threads"].get(str(args.pid), {})
                if main_thread.get("cpu_percent", 0) > 0:
                    main_idle_since = now
                record["main_thread_idle_seconds"] = round(now - main_idle_since, 3)
                record["monitor_gap_seconds"] = round(elapsed, 3)
            if now >= next_display:
                record["display"] = {}
                if shutil.which("xset"):
                    record["display"]["power"] = command_output(["xset", "q"])
                if shutil.which("xprop"):
                    record["display"]["active_window"] = command_output(
                        ["xprop", "-root", "_NET_ACTIVE_WINDOW"]
                    )
                if shutil.which("xwininfo"):
                    record["display"]["viewer_window"] = command_output(
                        ["xwininfo", "-name", "TOPDON TC002C Duo"]
                    )
                next_display = now + 10
            write(record)
            previous = record
            time.sleep(max(0, 1 - (time.monotonic() - now)))
        write({"event": "duration_reached", "time": datetime.now(UTC).isoformat()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
