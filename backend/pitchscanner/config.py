"""Configuration. Everything comes from the environment; no secrets in the repo."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PITCHSCANNER_", env_file=".env", extra="ignore"
    )

    # --- storage -----------------------------------------------------------
    dsn: str = Field(
        default="postgresql://pitchscanner:pitchscanner@localhost:5432/pitchscanner",
        description="Postgres DSN. Ignored when store=memory.",
    )
    store: str = Field(
        default="postgres",
        description="'postgres' for the real thing, 'memory' for tests and the offline demo.",
    )
    fixture_dir: Path = Field(default=REPO_ROOT / "data" / "fixtures" / "statsbomb")

    # --- data --------------------------------------------------------------
    open_data_repo: str = "https://github.com/statsbomb/open-data.git"
    #: Pinned so every build ingests byte-identical input.
    open_data_commit: str = "0067cae166a56aa80b2ef18f61e16158d6a7359a"
    data_cache: Path = Field(default=REPO_ROOT / ".data" / "open-data")

    # --- encoding and search ----------------------------------------------
    encoding: str = "E1"
    index_kind: str = Field(default="hnsw", description="hnsw | ivfflat | exact")
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 80
    ivfflat_lists: int = 200
    ivfflat_probes: int = 20
    #: Ask the index for candidate_multiplier x k, because dedup and filters
    #: throw candidates away.
    candidate_multiplier: int = 5
    max_candidate_multiplier: int = 40
    #: Below this many filtered candidates, ANN recall collapses, so fall back
    #: to an exact scan of the filtered subset.
    exact_fallback_threshold: int = 2000
    mirror_augmentation: bool = True

    # --- outcomes ----------------------------------------------------------
    outcome_window_s: float = 15.0
    low_confidence_n: int = 30
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 20260925

    # --- api ---------------------------------------------------------------
    max_k: int = 100
    cache_size: int = 512
    rate_limit_per_minute: int = 120
    cors_origins: str = "http://localhost:5173"

    # --- natural-language filters -----------------------------------------
    fuzzy_threshold: int = 84
    llm_fallback: bool = False
    llm_model: str = "claude-opus-5"
    anthropic_api_key: str | None = None

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
