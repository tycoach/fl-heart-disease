"""Download the four UCI Heart Disease site files into data/raw/."""
import urllib.request
from pathlib import Path

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
FILES = [f"processed.{s}.data" for s in ("cleveland", "hungarian", "switzerland", "va")]

out = Path("data/raw")
out.mkdir(parents=True, exist_ok=True)
for name in FILES:
    dest = out / name
    if dest.exists():
        print(f"skip {name} (already present)")
        continue
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(BASE + name, dest)
print("done")
