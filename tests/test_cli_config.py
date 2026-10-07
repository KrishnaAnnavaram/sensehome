import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sensehome.cli import main
from sensehome.config import Settings


def test_generate_validate_evaluate(tmp_path, capsys):
    d = tmp_path / "syn"
    assert main(["generate", "--out", str(d), "--items", "60", "--users", "40", "--events", "10"]) == 0
    assert (d / "catalogue.json").is_file() and (d / "interactions.csv").is_file()
    assert not list(d.glob("*.png")) and not list(d.glob("*.wav"))  # no media files are written
    capsys.readouterr()
    assert main(["validate", "--data", str(d)]) == 0
    assert json.loads(capsys.readouterr().out)["items"] == 60
    assert main(["evaluate", "--data", str(d), "--no-ablations", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["model"] for r in rows} >= {"popularity", "bpr", "bpr+features"}


def test_train_and_recommend(tmp_path, capsys):
    d = tmp_path / "syn"
    main(["generate", "--out", str(d), "--items", "60", "--users", "40", "--events", "10"])
    out = tmp_path / "m" / "recs.json"
    assert main(["train", "--data", str(d), "--out", str(out), "--k", "5"]) == 0
    payload = json.loads(out.read_text())
    assert payload["k"] == 5 and len(payload["recommendations"]) == 40 and "theta" in payload["attention"]
    capsys.readouterr()
    assert main(["recommend", "--data", str(d), "--user", "u0001", "--k", "3"]) == 0
    text = capsys.readouterr().out
    assert "modality weights" in text and text.count("\n") >= 4
    assert main(["recommend", "--data", str(d), "--user", "nobody"]) == 1
    assert main(["recommend", "--data", str(d), "--user", "u0001", "--model", "bpr", "--k", "2"]) == 0


def test_missing_folder_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        main(["validate", "--data", str(tmp_path / "nothing")])


def test_default_folder_falls_back_to_synthetic(capsys):
    assert main(["validate"]) == 0
    assert json.loads(capsys.readouterr().out)["items"] == 240


def test_settings(monkeypatch):
    monkeypatch.setenv("SENSEHOME_K", "5")
    monkeypatch.setenv("SENSEHOME_IMAGE_ENCODER", "clip")
    s = Settings.from_env()
    assert s.k == 5 and s.image_encoder == "clip"
    monkeypatch.setenv("SENSEHOME_K", "0")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.setenv("SENSEHOME_K", "5")
    monkeypatch.setenv("SENSEHOME_IMAGE_ENCODER", "resnet")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_core_import_does_not_load_heavy_frameworks():
    code = "import sys, sensehome.cli, sensehome.evaluate; print(any(m in sys.modules for m in ('torch','tensorflow','transformers')))"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.strip() == "False"
