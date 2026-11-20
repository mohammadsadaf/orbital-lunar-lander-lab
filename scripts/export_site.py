"""Export the measured lab as a static site for GitHub Pages or Sites hosting."""

from pathlib import Path
import json
import shutil


ROOT = Path(__file__).resolve().parents[1]
KNOWN_RUNS = ["baseline-seed7", "time-seed7", "hover-seed7", "potential-seed7", "random"]


def export(destination=None):
    destination = Path(destination or ROOT / "docs")
    destination.mkdir(parents=True, exist_ok=True)
    web = ROOT / "rocketlab/web"
    for source in web.iterdir():
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    index = (destination / "index.html").read_text()
    index = index.replace('href="/style.css"', 'href="./style.css"')
    index = index.replace('<script src="/app.js"></script>', '<script>window.ORBITAL_HOSTED=true;</script><script src="./app.js"></script>')
    (destination / "index.html").write_text(index)
    (destination / "style.css").write_text((destination / "style.css").read_text().replace("url('/", "url('./"))
    (destination / ".nojekyll").touch()
    data = destination / "data"
    data.mkdir(exist_ok=True)
    listing = []
    for name in KNOWN_RUNS:
        source = ROOT / "runs" / name
        if not (source / "evaluation.json").exists():
            continue
        target = data / name
        target.mkdir(exist_ok=True)
        for filename in ["config.json", "progress.json", "evaluation.json", "replay.json", "episodes.csv"]:
            if (source / filename).exists():
                shutil.copy2(source / filename, target / filename)
        if not (target / "progress.json").exists():
            (target / "progress.json").write_text("{}")
        videos = []
        for video in (source / "videos").glob("*.mp4"):
            (target / "videos").mkdir(exist_ok=True)
            shutil.copy2(video, target / "videos" / video.name)
            videos.append(video.name)
        progress = json.loads((target / "progress.json").read_text())
        listing.append({"id":name,
                        "config":json.loads((source / "config.json").read_text()) if (source / "config.json").exists() else {},
                        "progress":{k:v for k,v in progress.items() if k!="history"},
                        "evaluation":json.loads((source / "evaluation.json").read_text()),
                        "has_model":(source / "best.zip").exists(),
                        "has_replay":(target / "replay.json").exists(),
                        "error":None,"videos":videos})
    (data / "runs.json").write_text(json.dumps({"runs":listing,"job":None},allow_nan=False))
    print(f"Exported {len(listing)} measured experiments to {destination}")
    return destination


if __name__ == "__main__":
    export()
