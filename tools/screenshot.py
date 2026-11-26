import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "docs" / "_build" / "html"
SHOTS = ROOT / "docs" / "_build" / "shots"
CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
]


def find_browser():
    for candidate in CANDIDATES:
        if pathlib.Path(candidate).is_file():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return None


def main(arguments):
    if not arguments:
        print("usage: make shot PAGE=intro/neural_networks [HEIGHT=3000]")
        return 2
    page = arguments[0].removesuffix(".md").removesuffix(".html")
    height = arguments[1] if len(arguments) > 1 else "3000"
    source = BUILD / f"{page}.html"
    if not source.is_file():
        print(f"{source} does not exist. Run 'make check' first and give the page as section/name.")
        return 1
    browser = find_browser()
    if not browser:
        print("No Chrome, Chromium or Edge found. Run 'make serve' and look at the page in a browser instead.")
        return 1
    SHOTS.mkdir(parents=True, exist_ok=True)
    target = SHOTS / (page.replace("/", "_") + ".png")
    subprocess.run(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            f"--window-size=1300,{height}",
            "--virtual-time-budget=8000",
            f"--screenshot={target}",
            source.as_uri(),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not target.is_file():
        print("The browser did not produce a screenshot.")
        return 1
    print(target)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
