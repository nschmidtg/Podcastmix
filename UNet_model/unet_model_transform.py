# full assembly of the sub-parts to form the complete net
import math
import torch
import torchaudio
from unet_parts_transform import *
from asteroid.models import BaseModel
from asteroid.filterbanks.enc_dec import Filterbank, Encoder, Decoder
from asteroid.filterbanks import STFTFB
from asteroid.filterbanks import make_enc_dec


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

        # declare layers
        self.down1 = down(1, 16, self.kernel_size_c, self.stride_c)
        self.down2 = down(16, 32, self.kernel_size_c, self.stride_c)
        self.down3 = down(32, 64, self.kernel_size_c, self.stride_c)
        self.down4 = down(64, 128, self.kernel_size_c, self.stride_c)


        self.up1 = up(128 + 64, 64, self.kernel_size_d, self.stride_d, (1,0))
        self.up2 = up(64 + 32, 32, self.kernel_size_d, self.stride_d, (1,0))
        self.up3 = up(32 + 16, 16, self.kernel_size_d, self.stride_d, (1,0))
        self.up4 = up(16 + 1, 1, self.kernel_size_d, self.stride_d, (1,0))
        self.sigmoid = torch.nn.Sigmoid()

        # Create STFT/iSTFT pair in one line
        self.stft, self.istft = make_enc_dec('stft', n_filters=1024, kernel_size=self.fft_size, stride=self.hop_size)



    def forward(self, x_in):
        # compute normalized spectrogram
        X_in = self.stft(x_in)
        print("X_in:", X_in.shape)
        # add channels dimension
        X = X_in.unsqueeze(1)
        print("X:", X.shape)
        # first down layer
        X1 = self.down1(X)
        print("X1:", X1.shape)
        # second down layer
        X2 = self.down2(X1)
        print("X2:", X2.shape)
        # third down layer
        X3 = self.down3(X2)
        print("X3:", X3.shape)
        # fourth down layer
        X4 = self.down4(X3)
        print("X4:", X4.shape)


        # first up layer
        X3 = self.up1(X3, X4)
        print("X3 after 1 deconv:", X3.shape)
        # second up layer
        X2 = self.up2(X2, X3)
        print("X2 after 2 deconv:", X2.shape)
        # third up layer
        X1 = self.up3(X1, X2)
        print("X1 after 3 deconv:", X1.shape)
        # fourth up layer
        X = self.up4(X, X1)
        print("X after 1 deconv:", X.shape)
        
        # activation function
        X = self.sigmoid(X)
        print("X after sigmoid:", X.shape)
        # remove channels dimension:
        X = X.squeeze(1)
        print("X after squeeze:", X.shape)
        # use mask to separate speech from mix
        speech = X_in * X
        print("speech:", speech.shape)
        # use the opposite of the mask to separate music from mix
        music = X_in * (1 - X)
        print("music:", music.shape)
        # use GriffinLim to compute wav from normalized spectrogram
        speech_out = self.istft(speech)
        music_out = self.istft(music)
        print("speech_out:", speech_out.shape)
        print("music_out:", music_out.shape)
        # remove additional dimention
        speech_out = speech_out.squeeze(1)
        music_out = music_out.squeeze(1)

        # add both sources to a tensor to return them
        T_data = torch.stack([speech_out, music_out], dim=1)
        print("T_data:", T_data)
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
