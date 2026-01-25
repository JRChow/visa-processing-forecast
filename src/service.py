import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple

from src.model import VisaSurvivalModel
from src.refresh import refresh_data


def _bucket_major(value: str) -> str:
    value = str(value).lower()
    import re

    cs_pat = r"\b(cs|cse|eecs|computer\s+science|soft|ai|machine\s+learning|ml|nlp|data|algorithm|vision)\b"
    if re.search(cs_pat, value):
        return "CS"

    ece_pat = r"\b(ee|ece|elect|robot|circuit|micro|nano|semiconductor)\b"
    if re.search(ece_pat, value):
        return "ECE"

    stem_pat = r"\b(bio|chem|phys|mater|math|stat|mech|civil|aero|nuclear|health|med)\b"
    if re.search(stem_pat, value):
        return "STEM"

    return "Other"


def _read_mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


def _now_ts() -> float:
    return time.time()


@dataclass
class ModelStore:
    data_path: str
    cache_dir: str
    months_back: int
    refresh_ttl_seconds: int
    refresh_min_interval: int
    calibration_factor: float
    model: VisaSurvivalModel | None = None
    model_mtime: float = 0.0
    params_cache: Dict[Tuple[str, str, str], Tuple[Tuple[float, ...], float]] = field(default_factory=dict)
    last_refresh_attempt: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def ensure_model(self) -> None:
        with self.lock:
            mtime = _read_mtime(self.data_path)
            if self.model is None or mtime > self.model_mtime:
                if mtime == 0:
                    self._refresh_locked(force=True)
                    mtime = _read_mtime(self.data_path)
                self.model = VisaSurvivalModel(raw_data_path=self.data_path)
                self.model_mtime = mtime
                self.params_cache.clear()

    def should_refresh(self) -> bool:
        mtime = _read_mtime(self.data_path)
        if mtime == 0:
            return True
        return (_now_ts() - mtime) > self.refresh_ttl_seconds

    def maybe_refresh_background(self) -> None:
        if not self.should_refresh():
            return
        if _now_ts() - self.last_refresh_attempt < self.refresh_min_interval:
            return
        self.last_refresh_attempt = _now_ts()
        thread = threading.Thread(target=self.refresh_now, daemon=True)
        thread.start()

    def refresh_now(self) -> int:
        with self.lock:
            return self._refresh_locked(force=True)

    def _refresh_locked(self, force: bool = False) -> int:
        if not force and not self.should_refresh():
            return 0
        count = refresh_data(
            data_path=self.data_path,
            cache_dir=self.cache_dir,
            months_back=self.months_back,
            recent_window_days=90,
        )
        self.model = VisaSurvivalModel(raw_data_path=self.data_path)
        self.model_mtime = _read_mtime(self.data_path)
        self.params_cache.clear()
        return count

    def get_params(self, consulate: str, visa_type: str, major: str) -> Tuple[float, ...]:
        if self.model is None:
            raise RuntimeError("Model not initialized")

        key = (consulate, visa_type, major)
        cached = self.params_cache.get(key)
        if cached:
            return cached[0]

        params = self.model.fit_aft(
            consulate=consulate,
            visa_type=visa_type,
            major_bucket=major,
            tau=45,
            ghost_decay=90,
        )
        self.params_cache[key] = (params, _now_ts())
        return params

    def predict(self, consulate: str, visa_type: str, major: str, t0: int) -> dict:
        if self.model is None:
            raise RuntimeError("Model not initialized")

        params = self.get_params(consulate, visa_type, major)
        forecast = self.model.predict_conditional(params, t0, calibration_factor=self.calibration_factor)
        return forecast

    @staticmethod
    def bucket_major(value: str) -> str:
        return _bucket_major(value)


def build_store() -> ModelStore:
    data_path = os.getenv("VISA_DATA_PATH", "data/raw_data.csv")
    cache_dir = os.getenv("VISA_CACHE_DIR", "cache")
    months_back = int(os.getenv("VISA_MONTHS_BACK", "18"))
    refresh_ttl = int(os.getenv("VISA_REFRESH_TTL_SECONDS", str(6 * 3600)))
    refresh_min = int(os.getenv("VISA_REFRESH_MIN_INTERVAL", str(60 * 60)))
    calibration = float(os.getenv("VISA_CALIBRATION_FACTOR", "1.5"))

    return ModelStore(
        data_path=data_path,
        cache_dir=cache_dir,
        months_back=months_back,
        refresh_ttl_seconds=refresh_ttl,
        refresh_min_interval=refresh_min,
        calibration_factor=calibration,
    )
