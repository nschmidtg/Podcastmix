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
        self.down5 = down(128, 256, self.kernel_size_c, self.stride_c)
        self.down6 = down(256, 512, self.kernel_size_c, self.stride_c)


        self.up1 = up(512, 256, self.kernel_size_d, self.stride_d, (0,1), 1)
        self.up2 = up(256, 128, self.kernel_size_d, self.stride_d, (0,1), 2)
        self.up3 = up(128, 64, self.kernel_size_d, self.stride_d, (0,1), 3)
        self.up4 = up(64, 32, self.kernel_size_d, self.stride_d, (1,1), 4)
        self.up5 = up(32, 16, self.kernel_size_d, self.stride_d, (0,0), 5)
        self.up6 = up(16, 1, self.kernel_size_d, self.stride_d, (1,0), 6)

        # last activation layer
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
        # 5th down layer
        X5 = self.down5(X4)
        print("X5:", X5.shape)
        # 6th down layer
        X6 = self.down6(X5)
        print("X6:", X6.shape)


        # 1 up layer
        X5 = self.up1(X5, X6)
        print("X5 after 1 deconv:", X5.shape)
        # 2 up layer
        X4 = self.up2(X4, X5)
        print("X4 after 2 deconv:", X4.shape)
        # 3 up layer
        X3 = self.up3(X3, X4)
        print("X3 after 3 deconv:", X3.shape)
        # 4 up layer
        X2 = self.up4(X2, X3)
        print("X2 after 4 deconv:", X2.shape)
        # 5 up layer
        X1 = self.up5(X1, X2)
        print("X1 after 5 deconv:", X1.shape)
        # 6 up layer
        X = self.up6(X, X1)
        print("X after 6 deconv:", X.shape)
        
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
