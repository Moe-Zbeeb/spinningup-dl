import pathlib

import torch


def load_model(path, device="cpu"):
    from spinup_dl.models import MODEL_CLASSES

    path = pathlib.Path(path)
    if path.is_dir():
        path = path / "pyt_save" / "model.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MODEL_CLASSES[checkpoint["model"]](**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    model.extra = checkpoint.get("extra", {})
    return model
