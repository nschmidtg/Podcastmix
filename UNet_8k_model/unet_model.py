# full assembly of the sub-parts to form the complete net
import math
import torch
import torchaudio
from unet_parts import *
from asteroid.models import BaseModel
from asteroid.filterbanks.enc_dec import Filterbank, Encoder, Decoder
from asteroid.filterbanks import STFTFB
from asteroid.filterbanks import make_enc_dec


class UNet(BaseModel):
    #def __init__(self, n_channels, n_classes, bilinear=True):
    def __init__(self, sample_rate, fft_size, hop_size, window_size, kernel_size, stride):
        super(UNet, self).__init__()
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.kernel_size = kernel_size
        self.stride = stride


        # agregar el batchnorm al input

        # declare layers

        # input batch normalization
        self.input_layer = input_layer(1)

        # down blocks
        self.down1 = down(1, 16, self.kernel_size, self.stride)
        # self.down2 = down(16, 32, self.kernel_size, self.stride)
        # self.down3 = down(32, 64, self.kernel_size, self.stride)
        # self.down4 = down(64, 128, self.kernel_size, self.stride)
        # self.down5 = down(128, 256, self.kernel_size, self.stride)
        # self.down6 = down(256, 512, self.kernel_size, self.stride)

        # up blocks
        # self.up1 = up(512, 256, self.kernel_size, self.stride, (0,1), 1)
        # self.up2 = up(256, 128, self.kernel_size, self.stride, (0,1), 2)
        # self.up3 = up(128, 64, self.kernel_size, self.stride, (0,1), 3)
        # self.up4 = up(64, 32, self.kernel_size, self.stride, (0,1), 4)
        # self.up5 = up(32, 16, self.kernel_size, self.stride, (0,1), 5)
        self.last_layer = last_layer(16, 1, self.kernel_size, self.stride, (1, 0))

        # Create STFT/iSTFT pair in one line
        # self.stft, self.istft = make_enc_dec(
        #     'stft',
        #     n_filters=self.fft_size,
        #     kernel_size=self.window_size,
        #     stride=self.hop_size,
        #     sample_rate=self.sample_rate,
        #     output_padding = self.window_size // 2
        # )



    def forward(self, x_in):
        print("in:", x_in.shape)
        # compute normalized spectrogram
        X_in = torch.stft(x_in, self.fft_size, self.hop_size, window=torch.hann_window(self.window_size, device=x_in.device), return_complex=True, onesided=False)
        print("X_in after stft", X_in.shape)
        X_in = torch.abs(X_in)
        phase = torch.angle(X_in)
        # torch.random()
        # agregar matriz random para el profiling

        # add channels dimension
        X = X_in.unsqueeze(1)
        # X = X_in
        print("X:", X.shape)

        X = self.input_layer(X)

        # first down layer
        X1 = self.down1(X)
        print("X1:", X1.shape)

        # second down layer
        # X2 = self.down2(X1)
        # print("X2:", X2.shape)

        # third down layer
        # X3 = self.down3(X2)
        # print("X3:", X3.shape)

        # # fourth down layer
        # X4 = self.down4(X3)
        # print("X4 down4:", X4.shape)

        # # 5 down layer
        # X5 = self.down5(X4)
        # print("X5 down5:", X5.shape)

        # # 6 down layer
        # X6 = self.down6(X5)
        # print("X6 down6:", X6.shape)


        # # first up layer
        # X5 = self.up1(X5, X6)
        # print("X5 up1:", X5.shape)

        # # 2 up layer
        # X4 = self.up2(X4, X5)
        # print("X4 up2:", X4.shape)

        # # 3 up layer
        # X3 = self.up3(X3, X4)
        # print("X3 up3:", X3.shape)

        # # 4 up layer
        # X2 = self.up4(X2, X3)
        # print("X2 up4:", X2.shape)

        # # 5 up layer
        # X1 = self.up5(X1, X2)
        # print("X1 up5:", X1.shape)

        # last up layer (no concat after transposed conv)
        X = self.last_layer(X1)
        print("X last_layer:", X.shape)

        # remove channels dimension:
        X = X.squeeze(1)

        # use mask to separate speech from mix
        speech = X_in * X

        # use the complement of the mask to separate music from mix
        # music = X_in * (1 - X)

        # use ISTFT to compute wav from normalized spectrogram
        print("speech", speech.shape)
        print("phase", phase.shape)
        
        catted = torch.cat((speech.unsqueeze(1), phase.unsqueeze(1)), dim=1)

        print("catted", catted.shape)
        permuted = catted.permute(0,2,3,1).contiguous()
        print("permuted", permuted.shape)
        speech_as_complex = torch.view_as_complex(permuted)
        print("speech_as_complex", speech_as_complex.shape)
        speech_out = torch.istft(speech_as_complex.permute(0,1,2), self.fft_size, hop_length=self.hop_size, window=torch.hann_window(self.window_size, device=speech_as_complex.device), return_complex=True, onesided=False)
        # music_out = self.istft(music)

        # remove additional dimention
        speech_out = speech_out.squeeze(1)
        music_out = x_in - speech_out
        # music_out = music_out.squeeze(1)

        print("speech_out:", speech_out.shape)
        print("music_out:", music_out.shape)
        # add both sources to a tensor to return them
        T_data = torch.stack([speech_out, music_out], dim=1)

        # # write the sources to disk to check progress
        # torchaudio.save('speech0.wav', speech_out[0].unsqueeze(0).cpu(), sample_rate=8192)
        # torchaudio.save('music0.wav', music_out[0].unsqueeze(0).cpu(), sample_rate=8192)

        return T_data

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        model_args = {
            "fft_size": self.fft_size,
            "hop_size": self.hop_size,
            "window_size": self.window_size,
            "kernel_size": self.kernel_size,
            "stride": self.stride
        }
        return model_args
