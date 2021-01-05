from PodcastMix import PodcastMix
from asteroid.engine import System
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import soundfile as sf
import os
import numpy as np
import random

train_loader, val_loader = PodcastMix.loaders_from_mini(task="linear_mono", batch_size=2)

"""# Train the network"""
# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer

# We train the same model architecture that we used for inference above.
from asteroid import DPRNNTasNet

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper

# Tell DPRNN that we want to separate to 2 sources.
model = DPRNNTasNet(n_src=2)

# PITLossWrapper works with any loss function.
loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

optimizer = optim.Adam(model.parameters(), lr=1e-3)

system = System(model, optimizer, loss, train_loader, val_loader)

# Train for 1 epoch using a single GPU. If you're running this on Google Colab,
# be sure to select a GPU runtime (Runtime → Change runtime type → Hardware accelarator).
trainer = Trainer(max_epochs=1, gpus=1)
trainer.fit(system)

# !pip install librosa --quiet

import librosa

# Or simply a file name:
model.separate("/content/augmented_dataset/val/linear_mono/Ben-Carrigan-We-ll-Talk-About-It-All-Tonight-stem-mp4_1993-147964-0004_6345-93302-0016.wav", resample=True)

from IPython.display import display, Audio

display(Audio("/content/augmented_dataset/val/linear_mono/Ben-Carrigan-We-ll-Talk-About-It-All-Tonight-stem-mp4_1993-147964-0004_6345-93302-0016.wav"))
display(Audio("/content/augmented_dataset/val/linear_mono/Ben-Carrigan-We-ll-Talk-About-It-All-Tonight-stem-mp4_1993-147964-0004_6345-93302-0016_est1.wav"))
display(Audio("/content/augmented_dataset/val/linear_mono/Ben-Carrigan-We-ll-Talk-About-It-All-Tonight-stem-mp4_1993-147964-0004_6345-93302-0016_est2.wav"))
