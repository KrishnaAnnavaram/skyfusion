"""Site and experiment settings from environment variables."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _env(name: str, default: str) -> str:
    v = os.environ.get("SKYFUSION_" + name, "")
    return v.strip() or default


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    lat: float = Field(32.8998, ge=-90, le=90)
    lon: float = Field(-97.0403, ge=-180, le=180)
    cache_dir: str = "cache"
    train_end: int = 2019  # last year of the train split
    val_end: int = 2021  # last year of the validation split; later years are test
    horizon: int = Field(24, ge=1, le=72)
    window: int = Field(24, ge=1, le=168)
    seed: int = 7

    @model_validator(mode="after")
    def _years(self) -> "Settings":
        if self.val_end <= self.train_end:
            raise ValueError("SKYFUSION_VAL_END must be later than SKYFUSION_TRAIN_END")
        return self

    @property
    def lst_offset_hours(self) -> int:
        """NASA POWER local standard time: UTC + round(lon / 15) hours (no daylight saving)."""
        return int(round(self.lon / 15))

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            lat=float(_env("LAT", "32.8998")),
            lon=float(_env("LON", "-97.0403")),
            cache_dir=_env("CACHE_DIR", "cache"),
            train_end=int(_env("TRAIN_END", "2019")),
            val_end=int(_env("VAL_END", "2021")),
            horizon=int(_env("HORIZON", "24")),
            window=int(_env("WINDOW", "24")),
            seed=int(_env("SEED", "7")),
        )
