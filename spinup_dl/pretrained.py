import hashlib
import json
import pathlib
import sys
import urllib.request

from spinup_dl.user_config import DEFAULT_DATA_DIR
from spinup_dl.utils.load import load_model

REGISTRY_PATH = pathlib.Path(__file__).resolve().parent / "pretrained.json"
BASE_URL = "https://github.com/Moe-Zbeeb/spinningup-dl/releases/download/{tag}/{file}"


def registry():
    return json.loads(REGISTRY_PATH.read_text())


def list_pretrained():
    return sorted(registry()["models"])


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_pretrained(name, cache_dir=None, base_url=None):
    reg = registry()
    if name not in reg["models"]:
        raise ValueError(f"unknown pre-trained model '{name}'; choose from {list_pretrained()}")
    entry = reg["models"][name]
    cache = pathlib.Path(cache_dir or DEFAULT_DATA_DIR / "pretrained")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / entry["file"]
    if path.exists() and _sha256(path) == entry["sha256"]:
        return path
    url = (base_url or BASE_URL).format(tag=reg["tag"], file=entry["file"])
    print(f"Downloading {url}", flush=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    digest = _sha256(tmp)
    if digest != entry["sha256"]:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch for {entry['file']}: expected {entry['sha256']}, got {digest}")
    tmp.replace(path)
    return path


def load_pretrained(name, device="cpu", cache_dir=None, base_url=None):
    return load_model(download_pretrained(name, cache_dir, base_url), device=device)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    reg = registry()
    if argv:
        for name in argv:
            print(download_pretrained(name))
        return
    print(f"Pre-trained models (release {reg['tag']}):")
    for name in list_pretrained():
        entry = reg["models"][name]
        print(f"  {name:<24}{entry['result']}")


if __name__ == "__main__":
    main()
