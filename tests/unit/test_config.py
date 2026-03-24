"""Tests for config and storage utilities."""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from systematic_regime_trading.utils.config import (
    load_config,
    get_path,
    get_project_root,
)
from systematic_regime_trading.utils.paths import (
    get_data_dir,
    get_processed_dir,
    get_raw_dir,
    get_cache_dir,
)
from systematic_regime_trading.data.storage import (
    save_parquet,
    load_parquet,
    exists,
)


class TestLoadConfig:
    def test_loads_data_config(self):
        cfg = load_config("data")
        assert "universe" in cfg
        assert "cleaning" in cfg
        assert "storage" in cfg

    def test_loads_models_config(self):
        cfg = load_config("models")
        assert cfg["hmm"]["n_states"] == 5
        assert cfg["garch"]["distribution"] == "t"
        assert cfg["kmeans"]["n_clusters"] == 3

    def test_raises_on_missing_config(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config")


class TestPaths:
    def test_project_root_exists(self):
        root = get_project_root()
        assert root.exists()
        assert (root / "pyproject.toml").exists()

    def test_data_dirs_exist(self):
        assert get_data_dir().exists()
        assert get_processed_dir().exists()
        assert get_raw_dir().exists()

    def test_get_path_resolves(self):
        path = get_path("configs/data.yaml")
        assert path.exists()


class TestParquetStorage:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        # Monkey-patch the processed dir to use temp
        import systematic_regime_trading.data.storage as storage_mod
        monkeypatch.setattr(
            storage_mod, "_get_processed_dir", lambda: tmp_path,
        )

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        save_parquet(df, "test_roundtrip")
        loaded = load_parquet("test_roundtrip")

        pd.testing.assert_frame_equal(df, loaded)

    def test_exists_check(self, tmp_path, monkeypatch):
        import systematic_regime_trading.data.storage as storage_mod
        monkeypatch.setattr(
            storage_mod, "_get_processed_dir", lambda: tmp_path,
        )

        assert not exists("nonexistent")

        df = pd.DataFrame({"x": [1]})
        save_parquet(df, "exists_test")
        assert exists("exists_test")

    def test_load_missing_raises(self, tmp_path, monkeypatch):
        import systematic_regime_trading.data.storage as storage_mod
        monkeypatch.setattr(
            storage_mod, "_get_processed_dir", lambda: tmp_path,
        )

        with pytest.raises(FileNotFoundError):
            load_parquet("does_not_exist")
