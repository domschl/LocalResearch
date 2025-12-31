import socket
import json
import os
import logging
from operator import attrgetter
from typing import TypedDict, cast

class PerfRec(TypedDict):
    host: str
    task: str
    backend: str
    device: str
    timing: float
    unit: str

class PerfStats:
    def __init__(self, state_dir: str, global_state_dir: str):
        self.hostname: str = socket.gethostname()
        self.perf: list[PerfRec] = []
        self.log: logging.Logger = logging.getLogger("PerfStats")
        self.state_dir:str = state_dir
        self.global_state_dir:str = global_state_dir
        self.perf_file:str = os.path.join(self.state_dir, "perf_stats.json")
        self.perf_glob_file:str = os.path.join(self.global_state_dir, "perf_stats.json")

    def _read_perf_file(self):
        self.perf = []
        if os.path.exists(self.perf_glob_file):
            try:
                with open(self.perf_glob_file, "r") as f:
                    self.perf = cast(list[PerfRec], json.load(f))
            except Exception as e:
                self.log.error(f"Failed to read perf file: {e}")
                self.perf = []
        elif os.path.exists(self.perf_file):
            try:
                with open(self.perf_file, "r") as f:
                    self.perf = cast(list[PerfRec], json.load(f))
            except Exception as e:
                self.log.error(f"Failed to read perf file: {e}")
                self.perf = []
        else:
            self.perf = []
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

    def add_perf(self, task: str, backend: str, device: str, timing: float, unit: str):
        self._read_perf_file()
        host = self.hostname
        perf_n: PerfRec = PerfRec(host=host, task=task, backend=backend, device=device, timing=timing, unit=unit)
        new_rec = True
        for index, perf_i in enumerate(self.perf):
            if perf_i['host'] == perf_n['host'] and perf_i['task'] == perf_n['task'] and perf_i['backend'] == perf_n['backend'] and perf_i['device'] == perf_n['device']:
                new_rec = False
                timing = (perf_i['timing'] * 4 + perf_n['timing']) / 5.0
                self.perf[index]['timing'] = timing
                break
        if new_rec is True:
            self.perf.append(perf_n)
        self.perf = sorted(self.perf, key=lambda x: (x['task'], x['timing']))
        self._write_perf_file()

    def get_perf(self):
        self._read_perf_file()
        return self.perf

    def sync_perf(self):
        self._read_perf_file()
        self._write_perf_file()
    
