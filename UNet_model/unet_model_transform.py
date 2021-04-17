# full assembly of the sub-parts to form the complete net
import math
import torch
import torchaudio
from unet_parts_transform import *
from asteroid.models import BaseModel

class UNet(BaseModel):
    #def __init__(self, n_channels, n_classes, bilinear=True):
    def __init__(self, segment, sample_rate, fft_size, hop_size, window_size, kernel_size_c, stride_c, kernel_size_d, stride_d):
        super(UNet, self).__init__()
        self.segment = segment
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.kernel_size_c = kernel_size_c
        self.stride_c = stride_c
        self.kernel_size_d = kernel_size_d
        self.stride_d = stride_d

        # create a hamming window for the STFT
        self.window = torch.hamming_window(self.window_size).cuda()

        # declare layers
        self.down1 = down(1, 16, self.kernel_size_c, self.stride_c)
        self.up1 = up(16 + 1, 1, self.kernel_size_d, self.stride_d)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_in):
        # compute normalized spectrogram
        X_in = torchaudio.transforms.Spectrogram(
            self.fft_size,
            win_length=self.window_size,
            hop_length=self.hop_size,
            window_fn=torch.hamming_window,
            normalized=True
        ).cuda()(x_in)

        # add channels dimension
        X = X_in.unsqueeze(1)

        # first down layer
        X1 = self.down1(X)

        # first up layer
        X = self.up1(X, X1)

        # activation function
        X = self.sigmoid(X)

        # remove channels dimension:
        X = X.squeeze(1)

        # use mask to separate speech from mix
        speech = X_in * X

        # use the opposite of the mask to separate music from mix
        music = X_in * (1 - X)

        # use GriffinLim to compute wav from normalized spectrogram
        speech_out = torchaudio.transforms.GriffinLim(
            self.fft_size,
            win_length=self.window_size,
            hop_length=self.hop_size,
            window_fn=torch.hamming_window,
            normalized=True
        ).cuda()(speech)
        music_out = torchaudio.transforms.GriffinLim(
            self.fft_size,
            win_length=self.window_size,
            hop_length=self.hop_size,
            window_fn=torch.hamming_window,
            normalized=True
        ).cuda()(music)

        # add both sources to a tensor to return them
        T_data = torch.stack([speech_out, music_out], dim=1)

        # # write the sources to disk to check progress
        # torchaudio.save('speech0.wav', speech_out[0].unsqueeze(0).cpu(), sample_rate=8192)
        # torchaudio.save('music0.wav', music_out[0].unsqueeze(0).cpu(), sample_rate=8192)
        
        return T_data

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        model_args = {
            "segment": self.segment,
            "sample_rate": self.sample_rate,
            "fft_size": self.fft_size,
            "hop_size": self.hop_size,
            "window_size": self.window_size,
            "kernel_size_c": self.kernel_size_c,
            "stride_c": self.stride_c,
            "kernel_size_d": self.kernel_size_d,
            "stride_d": self.stride_d
        }
        return model_args
