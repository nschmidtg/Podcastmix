# full assembly of the sub-parts to form the complete net
import math
import torch
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
import torchaudio
from openunmix_parts import *
from asteroid.models import BaseModel
from asteroid.filterbanks.enc_dec import Filterbank, Encoder, Decoder
from asteroid.filterbanks import STFTFB
from asteroid.filterbanks import make_enc_dec


class OpenUnmix(BaseModel):
    #def __init__(self, n_channels, n_classes, bilinear=True):
    def __init__(self, nb_bins, window_size, hop_size):
        super(OpenUnmix, self).__init__()
        
        self.nb_bins = nb_bins
        self.hop_size = hop_size
        self.window_size = window_size
        self.nb_layers = 3
        self.hidden_size = 512
        self.self.nb_channels = 2

        # declare layers
        self.nb_output_bins = nb_bins

        self.fc1 = Linear(self.nb_bins * self.nb_channels, self.hidden_size, bias=False)

        self.bn1 = BatchNorm1d(self.hidden_size)

        # bidirectional
        self.lstm_hidden_size = self.hidden_size // 2

        self.lstm = LSTM(
            input_size=self.hidden_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.nb_layers,
            bidirectional=True,
            batch_first=False,
            dropout=0.4,
        )

        fc2_hiddensize = self.hidden_size * 2
        self.fc2 = Linear(in_features=fc2_hiddensize, out_features=self.hidden_size, bias=False)

        self.bn2 = BatchNorm1d(self.hidden_size)

        self.fc3 = Linear(
            in_features=self.hidden_size,
            out_features=self.nb_output_bins * self.nb_channels,
            bias=False,
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins * self.nb_channels)

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[: self.nb_bins]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = Parameter(torch.ones(self.nb_output_bins).float())


        # Create STFT/iSTFT pair in one line
        self.stft, self.istft = make_enc_dec(
            'stft',
            n_filters=self.nb_bins,
            kernel_size=self.window_size,
            stride=self.hop_size
        )



    def forward(self, x_in):
        # compute normalized spectrogram
        X_in = self.stft(x_in)

        # add channels dimension
        X = X_in.unsqueeze(1)
        print("X:", X.shape)

        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)
        # get current spectrogram shape
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        # crop
        x = x[..., : self.nb_bins]
        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean
        x = x * self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer normx
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        x.permute(1, 2, 3, 0)
        
        # remove channels dimension:
        X = x.squeeze(1)

        # use mask to separate speech from mix
        speech = X_in * X

        # use the opposite of the mask to separate music from mix
        music = X_in * (1 - X)

        # use ISTFT to compute wav from normalized spectrogram
        speech_out = self.istft(speech)
        music_out = self.istft(music)

        # remove additional dimention
        speech_out = speech_out.squeeze(1)
        music_out = music_out.squeeze(1)

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
            "kernel_size": self.kernel_size,
            "stride": self.stride
        }
        return model_args
