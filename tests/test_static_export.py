import importlib.util
import json
from pathlib import Path


def test_static_archive_has_measured_runs_and_relative_assets(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("export_site", root / "scripts/export_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = module.export(tmp_path / "site")
    index = (destination / "index.html").read_text()
    assert 'window.ORBITAL_HOSTED=true' in index
    assert 'href="./style.css"' in index
    assert 'src="./app.js"' in index
    assert "url('./dm-sans.ttf')" in (destination / "style.css").read_text()
    listing = json.loads((destination / "data/runs.json").read_text())
    assert len(listing["runs"]) == 5
    assert listing["job"] is None
    for run in listing["runs"]:
        folder = destination / "data" / run["id"]
        assert (folder / "progress.json").is_file()
        assert (folder / "replay.json").is_file()
        evaluation = json.loads((folder / "evaluation.json").read_text())
        assert evaluation["episodes"] == 20
        assert evaluation["wrapped"] is False
        for video in run["videos"]:
            assert (folder / "videos" / video).is_file()
