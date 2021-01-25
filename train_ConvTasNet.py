from PodcastMix import PodcastMix
from asteroid.engine import System
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import soundfile as sf
import os
import numpy as np
import random
from IPython.display import display, Audio
import librosa

# define the task and load the dataset as a DataLoader
train_loader, val_loader = PodcastMix.loaders_from_mini(task="linear_mono", batch_size=2)

"""# Train the network"""
# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer
from asteroid import ConvTasNet

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper

# Tell DPRNN that we want to separate to 2 sources.
model = ConvTasNet(n_src=2, bn_chan=64, n_blocks=4)

# PITLossWrapper works with any loss function.
loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
system = System(model, optimizer, loss, train_loader, val_loader)

# Train for 1 epoch using a single GPU. If you're running this on Google Colab,
# be sure to select a GPU runtime (Runtime → Change runtime type → Hardware accelarator).
trainer = Trainer(max_epochs=150, gpus=1)
trainer.fit(system)

# get the test file from console
test_path = input("enter the path of the file to test:")

# use the model to separate a file
model.separate(test_path, resample=True, force_overwrite=True)

# display sounds
# display(Audio(test_path))
# display(Audio(test_path.split(".")[0] + '_est1.wav'))
# display(Audio(test_path.split(".")[0] + '_est2.wav'))
print(test_path.split(".")[0] + '_est1.wav')
print(test_path.split(".")[0] + '_est2.wav')
