import json

import pytest
import torch

import spinup_dl
from spinup_dl.run import main as run_main
from spinup_dl.utils.load import load_model
from spinup_dl.utils.plot import make_plots, read_progress


def logger_kwargs(tmp_path, name):
    return {"output_dir": str(tmp_path / name), "exp_name": name}


@pytest.mark.parametrize("name, kwargs", [
    ("mlp", {"hidden_sizes": (32,), "epochs": 1}),
    ("cnn", {"arch": "lenet", "epochs": 1}),
    ("cnn", {"arch": "resnet", "widths": (8, 16), "blocks_per_stage": 1, "epochs": 2, "lr_schedule": "cosine"}),
    ("vae", {"hidden_size": 32, "latent_dim": 4, "epochs": 1}),
    ("diffusion", {"base_channels": 8, "timesteps": 20, "epochs": 1, "num_samples": 4}),
])
def test_image_models_train_save_and_load(tmp_path, name, kwargs):
    trainer = getattr(spinup_dl, name)
    run = f"{name}_{kwargs.get('arch', 'run')}"
    model = trainer(dataset="fake", batch_size=32, device="cpu", logger_kwargs=logger_kwargs(tmp_path, run), **kwargs)
    out = tmp_path / run
    progress = read_progress(out / "progress.txt")
    assert progress["Epoch"].tolist() == [float(e) for e in range(1, kwargs["epochs"] + 1)]
    assert json.loads((out / "config.json").read_text())["exp_name"] == run
    loaded = load_model(out)
    state, loaded_state = model.state_dict(), loaded.state_dict()
    assert all(torch.equal(state[k].cpu(), loaded_state[k]) for k in state)


@pytest.mark.parametrize("name, kwargs", [
    ("rnn", {"cell": "gru", "hidden_size": 32, "num_layers": 1, "seq_len": 16}),
    ("rnn", {"cell": "lstm", "hidden_size": 32, "num_layers": 2, "seq_len": 16}),
    ("transformer", {"context_length": 16, "d_model": 32, "n_heads": 2, "n_layers": 1}),
])
def test_language_models_train_sample_and_load(tmp_path, name, kwargs):
    trainer = getattr(spinup_dl, name)
    run = f"{name}_{kwargs.get('cell', 'tf')}"
    trainer(dataset="fake", batch_size=8, epochs=1, steps_per_epoch=5, sample_length=20, device="cpu",
            logger_kwargs=logger_kwargs(tmp_path, run), **kwargs)
    out = tmp_path / run
    assert "TestPerplexity" in read_progress(out / "progress.txt")
    assert "epoch 1" in (out / "samples.txt").read_text()
    model = load_model(out)
    ids = torch.tensor([[0, 1, 2]])
    assert model.generate(ids, 5).shape == (1, 8)
    assert len(model.extra["chars"]) == model.config["vocab_size"]


def test_command_line_grid_and_plot(tmp_path):
    run_main(["mlp", "--dataset", "fake", "--hidden_sizes", "[16]", "[32]", "--epochs", "2", "--device", "cpu",
              "--exp_name", "grid", "--data_dir", str(tmp_path), "--seed", "0", "1"])
    runs = sorted(p.parent.name for p in tmp_path.rglob("progress.txt"))
    assert runs == ["grid_hidden_sizes16_s0", "grid_hidden_sizes16_s1", "grid_hidden_sizes32_s0", "grid_hidden_sizes32_s1"]
    figures = make_plots([tmp_path], values=["TestAccuracy"], output=str(tmp_path / "plot.png"))
    assert len(figures) == 1 and (tmp_path / "plot.png").exists()


def test_pretrained_download_verifies_checksum(tmp_path, monkeypatch):
    import hashlib
    from spinup_dl import pretrained

    run = "mlp_pre"
    spinup_dl.mlp(dataset="fake", hidden_sizes=(16,), epochs=1, device="cpu", logger_kwargs=logger_kwargs(tmp_path, run))
    weights = tmp_path / run / "pyt_save" / "model.pt"
    release = tmp_path / "release"
    release.mkdir()
    (release / "mlp_test.pt").write_bytes(weights.read_bytes())
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    reg = {"tag": "test", "models": {"mlp_test": {"file": "mlp_test.pt", "sha256": digest, "result": "test"}}}
    registry_path = tmp_path / "pretrained.json"
    registry_path.write_text(json.dumps(reg))
    monkeypatch.setattr(pretrained, "REGISTRY_PATH", registry_path)
    base_url = release.as_uri() + "/{file}"
    model = pretrained.load_pretrained("mlp_test", cache_dir=tmp_path / "cache", base_url=base_url)
    assert model.config["hidden_sizes"] == [16]
    reg["models"]["mlp_test"]["sha256"] = "0" * 64
    registry_path.write_text(json.dumps(reg))
    (tmp_path / "cache" / "mlp_test.pt").unlink()
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        pretrained.download_pretrained("mlp_test", cache_dir=tmp_path / "cache", base_url=base_url)
