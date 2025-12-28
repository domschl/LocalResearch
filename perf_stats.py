import socket
import json
import os
import logging

class PerfStats:
    def __init__(self, state_dir: str, global_state_dir: str):
        self.hostname: str = socket.gethostname()
        self.perf: dict[str, float] = {}
        self.log: logging.Logger = logging.getLogger("PerfStats")
        self.state_dir:str = state_dir
        self.global_state_dir:str = global_state_dir
        self.perf_file:str = os.path.join(self.state_dir, "perf_stats.json")
        self.perf_glob_file:str = os.path.join(self.global_state_dir, "perf_stats.json")

    def _read_perf_file(self):
        if os.path.exists(self.perf_glob_file):
            with open(self.perf_glob_file, "r") as f:
                self.perf = json.load(f)
        elif os.path.exists(self.perf_file):
            with open(self.perf_file, "r") as f:
                self.perf = json.load(f)
        else:
            self.perf = {}
            self._write_perf_file()

    def _write_perf_file(self):
        os.makedirs(os.path.dirname(self.perf_file), exist_ok=True)
        try:
            with open(self.perf_glob_file, "w") as f:
                json.dump(self.perf, f, indent=4)
        except Exception as _:
            self.log.info("Failed to write global perf file")
            pass
        try:
            with open(self.perf_file, "w") as f:
                json.dump(self.perf, f, indent=4)
        except Exception as e:
            self.log.info(f"Failed to write local perf file: {e}")

    def add_perf(self, key: str, value: float):
        self._read_perf_file()
        key = f"{self.hostname}_{key}"
        self.perf[key] = value
        self._write_perf_file()

    def get_perf(self):
        self._read_perf_file()
        return self.perf

    def sync_perf(self):
        self._read_perf_file()
        self._write_perf_file()
    
