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

        self.window = torch.hamming_window(self.window_size).cuda()
        self.number_of_samples_in_x = segment * sample_rate
        self.input_number_frames = math.floor(self.number_of_samples_in_x / hop_size) + 1

        self.down1 = down(1, 16, self.kernel_size_c, self.stride_c)
        self.up1 = up(16 + 1, 1, self.kernel_size_d, self.stride_d)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_in):
        # torchaudio.save('../x_0.wav', x[0].unsqueeze(0), sample_rate=8192)
        X_in = torchaudio.transforms.Spectrogram(self.fft_size, win_length=self.window_size, hop_length=self.hop_size, window_fn=torch.hamming_window, normalized=True).cuda()(x_in)
        # print("despues de stft", X_in.shape)
        X = X_in.unsqueeze(1)
        # print("despues de unsqueeze", X.shape)
        print("X antes de down1:", X)
        X1 = self.down1(X)
        # print("pasé el self.down1(X)")
        # print("X:", X.shape)
        # print("X1:", X1.shape)
        print("X1 antes de up:", X1)
        X = self.up1(X, X1)
        # print("pasé el self.up1(X, X1)")
        # print(X.shape)
        # print("X antes de sigmoid:", X)
        X = self.sigmoid(X)
        # print("pasé el sigmoid")
        # print(X.shape)

        # remove channels dimension:
        X = X.squeeze(1)

        print("X:", X.shape)
        print(X)
        print("X_in:", X_in.shape)
        print(X_in)
        print("1-X shape:", (1-X).shape)
        print("1-X", (1-X))
        # use mask to separate speech from mix:
        speech = X_in * X
        music = X_in * (1 - X)
        # print("speech after remove channel dimension:", speech.shape)
        # print("music after remove channel dimension:", music.shape)


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
        print("speech audio dp de istft:", speech_out.shape)
        print("music audio dp de istft:", music_out.shape)
        T_data = torch.stack([speech_out, music_out], dim=1)
        # T = torch.tensor(T_data)
        print("T", T_data.shape)
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
