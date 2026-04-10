import json

import pytest

from src.config_loader import load_config


def test_merges_defaults_with_file_values(tmp_path):
    config_path = tmp_path / "service.json"
    config_path.write_text(json.dumps({"port": 9090}), encoding="utf-8")

    loaded = load_config(str(config_path))

    assert loaded == {
        "host": "127.0.0.1",
        "port": 9090,
        "debug": False,
    }


def test_preserves_explicit_debug_setting(tmp_path):
    config_path = tmp_path / "service.json"
    config_path.write_text(
        json.dumps({"host": "0.0.0.0", "debug": True}),
        encoding="utf-8",
    )

    loaded = load_config(str(config_path))

    assert loaded["host"] == "0.0.0.0"
    assert loaded["debug"] is True
    assert loaded["port"] == 8080


def test_invalid_json_raises_value_error(tmp_path):
    config_path = tmp_path / "broken.json"
    config_path.write_text("{not-json}", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(str(config_path))
