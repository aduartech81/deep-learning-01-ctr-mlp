"""Medición de tiempo de pared, memoria pico (proceso + hijos, incluye la JVM de Spark) y
filas por segundo; registro de resultados y ajuste de la ley de Amdahl."""
from __future__ import annotations

import json
import platform
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psutil
from scipy.optimize import curve_fit


class PeakMemory:
    """Muestrea el RSS del proceso Python y de todos sus hijos (la JVM de Spark es hija)."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._proc = psutil.Process()

    def _rss(self):
        tot = 0
        try:
            tot = self._proc.memory_info().rss
            for ch in self._proc.children(recursive=True):
                try:
                    tot += ch.memory_info().rss
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
        return tot

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(self.interval)

    def __enter__(self):
        self.base = self._rss()
        self.peak = self.base
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self._rss())

    @property
    def peak_mb(self):
        return self.peak / 2**20

    @property
    def delta_mb(self):
        return (self.peak - self.base) / 2**20


@contextmanager
def measure(record: dict, stage: str, n_rows: int | None = None):
    """Uso: with measure(rec, 'train', n): ...  -> rec['train_s'], rec['train_peak_mb'], ..."""
    with PeakMemory() as pm:
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
    record[f"{stage}_s"] = dt
    record[f"{stage}_peak_mb"] = pm.peak_mb
    record[f"{stage}_delta_mb"] = pm.delta_mb
    if n_rows:
        record[f"{stage}_rows_per_s"] = n_rows / dt if dt > 0 else np.nan


def append_jsonl(path: Path, rec: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(rec, default=_json_default) + "\n")


def read_jsonl(path: Path):
    import pandas as pd
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def amdahl(p, s):
    """Aceleración de Amdahl con fracción secuencial s."""
    return 1.0 / (s + (1 - s) / p)


def fit_amdahl(cores, times) -> dict:
    cores = np.asarray(cores, float)
    times = np.asarray(times, float)
    t1 = times[cores == cores.min()].mean()
    S = t1 / times
    popt, pcov = curve_fit(amdahl, cores, S, p0=[0.3], bounds=(0, 1))
    s = float(popt[0])
    return {"fraccion_secuencial": s, "se": float(np.sqrt(pcov[0, 0])) if pcov.size else np.nan,
            "aceleracion_max_teorica": 1 / s if s > 0 else np.inf,
            "speedup": S.tolist(), "eficiencia": (S / cores).tolist()}


def hardware_info() -> dict:
    vm = psutil.virtual_memory()
    return {"sistema": platform.platform(), "procesador": platform.processor() or platform.machine(),
            "nucleos_fisicos": psutil.cpu_count(logical=False), "nucleos_logicos": psutil.cpu_count(),
            "ram_gb": round(vm.total / 2**30, 1), "python": platform.python_version()}
