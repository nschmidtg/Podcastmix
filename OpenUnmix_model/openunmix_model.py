# full assembly of the sub-parts to form the complete net
import math
import torch
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
import torchaudio
import torch.nn.functional as F
from asteroid.models import BaseModel
from asteroid.filterbanks.enc_dec import Filterbank, Encoder, Decoder
from asteroid.filterbanks import STFTFB
from asteroid.filterbanks import make_enc_dec


class OpenUnmix(BaseModel):
    #def __init__(self, n_channels, n_classes, bilinear=True):
    def __init__(self, sample_rate, nb_bins, window_size, hop_size):
        super(OpenUnmix, self).__init__()
        self.sample_rate = sample_rate
        self.nb_bins = nb_bins
        self.hop_size = hop_size
        self.window_size = window_size
        self.nb_layers = 3
        self.hidden_size = 512
        self.nb_channels = 1

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

        input_mean = torch.zeros(self.nb_bins)

        input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = Parameter(torch.ones(self.nb_output_bins).float())


        print(self.nb_bins, self.window_size, self.hop_size)
        # Create STFT/iSTFT pair in one line
        self.stft, self.istft = make_enc_dec(
            'stft',
            n_filters=1024,
            kernel_size=1024,
            stride=32,
            sample_rate=8000,
        )

        print("***************:", self.nb_bins)

    def forward(self, x_in):
        print("***************1:", self.nb_bins)
        print("input x_in:", x_in.shape)
        # compute normalized spectrogram
        X_in = self.stft(x_in)
        print("after stft", X_in.shape)
        aux_istft = self.istft(X_in)
        print("shape of aux_istst", aux_istft.shape)

        # add channels dimension
        X = X_in.unsqueeze(1)
        print("X:", X.shape)
        print("***************2:", self.nb_bins)
        # permute so that batch is last for lstm
        x = X.permute(3, 0, 1, 2)
        # get current spectrogram shape
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
        print("***************3:", self.nb_bins)
        mix = x.detach().clone()
        # mix = mix[..., : self.nb_bins]

        print("mix:", mix.shape)
        print("self.nb_bins:", self.nb_bins)
        # crop
        # x = x[..., : self.nb_bins]
        # shift and scale input to mean=0 std=1 (across all bins)
        x = x + self.input_mean
        x = x * self.input_scale

        print("x after mean", x.shape)
        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = x.reshape(-1, nb_channels * self.nb_bins)
        print("x after reshape", x.shape)
        x = self.fc1(x)
        print("after fc1", x.shape)
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
        mask = F.relu(x)
        speech = mask * mix
        music = (1 - mask) * mix
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        print("before permutation:", speech.shape)
        speech = speech.permute(1, 2, 3, 0)
        print("after permutation:", speech.shape)
        music = music.permute(1, 2, 3, 0)

        # remove channels dimension:
        speech = speech.squeeze(1)
        music = music.squeeze(1)
        print("speech_out", speech.shape)

        # use mask to separate speech from mix
        # speech = X_in * X

        # use the opposite of the mask to separate music from mix
        # music = X_in * (1 - X)

        # use ISTFT to compute wav from normalized spectrogram
        speech_out = self.istft(speech)
        music_out = self.istft(music)

        print("speech_out", speech_out.shape)

        # remove additional dimention
        speech_out = speech_out.squeeze(1)
        music_out = music_out.squeeze(1)
        print("speech_out", speech_out.shape)


        # add both sources to a tensor to return them
        T_data = torch.stack([speech_out, music_out], dim=1)

        # # write the sources to disk to check progress
        # torchaudio.save('speech0.wav', speech_out[0].unsqueeze(0).cpu(), sample_rate=8192)
        # torchaudio.save('music0.wav', music_out[0].unsqueeze(0).cpu(), sample_rate=8192)

        return T_data

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        model_args = {
            "sample_rate": self.sample_rate,
            "nb_bins": self.nb_bins,
            "hop_size": self.hop_size,
            "window_size": self.window_size
        }
        return model_args
