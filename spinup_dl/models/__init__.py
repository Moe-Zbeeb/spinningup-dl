from spinup_dl.models.cnn import LeNet, ResNet, cnn
from spinup_dl.models.diffusion import Diffusion, diffusion
from spinup_dl.models.mlp import MLP, mlp
from spinup_dl.models.rnn import CharRNN, rnn
from spinup_dl.models.transformer import Transformer, transformer
from spinup_dl.models.vae import VAE, vae

TRAINERS = {"mlp": mlp, "cnn": cnn, "rnn": rnn, "transformer": transformer, "vae": vae, "diffusion": diffusion}
MODEL_CLASSES = {cls.__name__: cls for cls in (MLP, LeNet, ResNet, CharRNN, Transformer, VAE, Diffusion)}
