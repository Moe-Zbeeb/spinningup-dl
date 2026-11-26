import json
import pathlib
import time

import torch

from spinup_dl.user_config import DEFAULT_DATA_DIR, FORCE_DATESTAMP


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, pathlib.Path):
        return str(obj)
    return repr(obj)


def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False):
    datestamp = datestamp or FORCE_DATESTAMP
    prefix = time.strftime("%Y-%m-%d_") if datestamp else ""
    relpath = prefix + exp_name
    if seed is not None:
        subfolder = (time.strftime("%Y-%m-%d_%H-%M-%S-") if datestamp else "") + f"{exp_name}_s{seed}"
        relpath = str(pathlib.Path(relpath) / subfolder)
    root = pathlib.Path(data_dir or DEFAULT_DATA_DIR)
    return {"output_dir": str(root / relpath), "exp_name": exp_name}


class Logger:
    def __init__(self, output_dir=None, output_fname="progress.txt", exp_name=None):
        if output_dir is None:
            output_dir = DEFAULT_DATA_DIR / "tmp" / f"experiment_{int(time.time())}"
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = open(self.output_dir / output_fname, "w")
        self.exp_name = exp_name
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.log(f"Logging data to {self.output_file.name}")

    def log(self, msg):
        print(msg, flush=True)

    def log_tabular(self, key, val):
        if self.first_row:
            self.log_headers.append(key)
        elif key not in self.log_headers:
            raise KeyError(f"'{key}' was not logged in the first row; log the same keys every epoch")
        if key in self.log_current_row:
            raise KeyError(f"'{key}' was already logged this epoch")
        self.log_current_row[key] = val

    def save_config(self, config):
        config = _jsonable({k: v for k, v in config.items() if k not in ("logger",)})
        if self.exp_name is not None:
            config["exp_name"] = self.exp_name
        text = json.dumps(config, indent=4, sort_keys=True)
        (self.output_dir / "config.json").write_text(text + "\n")
        self.log("Saving config:\n" + text)

    def save_model(self, model, extra=None):
        save_dir = self.output_dir / "pyt_save"
        save_dir.mkdir(exist_ok=True)
        checkpoint = {
            "model": type(model).__name__,
            "config": model.config,
            "state_dict": model.state_dict(),
            "extra": extra or {},
        }
        torch.save(checkpoint, save_dir / "model.pt")

    def dump_tabular(self):
        values = [self.log_current_row.get(k, "") for k in self.log_headers]
        width = max(len(k) for k in self.log_headers) + 2
        line = "-" * (width + 16)
        rows = [line]
        for key, val in zip(self.log_headers, values):
            shown = f"{val:.4g}" if isinstance(val, float) else str(val)
            rows.append(f"| {key:>{width - 2}} | {shown:>11} |")
        rows.append(line)
        self.log("\n".join(rows))
        if self.first_row:
            self.output_file.write("\t".join(self.log_headers) + "\n")
        self.output_file.write("\t".join(str(v) for v in values) + "\n")
        self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False

    def close(self):
        self.output_file.close()
