import os
import pathlib

DEFAULT_DATA_DIR = pathlib.Path(os.environ.get("SPINUP_DL_DATA", pathlib.Path(__file__).resolve().parent.parent / "data"))
DEFAULT_DATASET_DIR = DEFAULT_DATA_DIR / "datasets"
FORCE_DATESTAMP = False
