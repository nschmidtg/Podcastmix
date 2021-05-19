# full assembly of the sub-parts to form the complete net
import math
import torch
import torchaudio
from unet_parts import *
from asteroid.models import BaseModel

class UNet(BaseModel):
    #def __init__(self, n_channels, n_classes, bilinear=True):
    def __init__(self, sample_rate, fft_size, hop_size, window_size, kernel_size, stride):
        super(UNet, self).__init__(sample_rate=sample_rate)
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
        self.down2 = down(16, 32, self.kernel_size, self.stride)
        self.down3 = down(32, 64, self.kernel_size, self.stride)
        self.down4 = down(64, 128, self.kernel_size, self.stride)
        self.down5 = down(128, 256, self.kernel_size, self.stride)
        self.down6 = down(256, 512, self.kernel_size, self.stride)

        # up blocks
        self.up1 = up(512, 256, self.kernel_size, self.stride, (0,0), 1)
        self.up2 = up(256, 128, self.kernel_size, self.stride, (0,0), 2)
        self.up3 = up(128, 64, self.kernel_size, self.stride, (0,1), 3)
        self.up4 = up(64, 32, self.kernel_size, self.stride, (0,0), 4)
        self.up5 = up(32, 16, self.kernel_size, self.stride, (0,0), 5)
        self.last_layer = last_layer(16, 1, self.kernel_size, self.stride, (0, 0))



    def forward(self, x_in):
        # compute normalized spectrogram
        window = torch.hamming_window(self.window_size, device=x_in.get_device())
        X_in = torch.stft(x_in, self.fft_size, self.hop_size, window=window)
        print("X_in after stft", X_in.shape)
        real, imag = X_in.unbind(-1)
        complex_n = torch.cat((real.unsqueeze(1), imag.unsqueeze(1)), dim=1).permute(0,2,3,1).contiguous()
        r_i = torch.view_as_complex(complex_n)
        print("real", real.shape)
        print("imag", imag.shape)
        phase = torch.angle(r_i)
        X_in = torch.sqrt(real**2 + imag**2)

        # add channels dimension
        X = X_in.unsqueeze(1)
        # X = X_in
        print("X:", X.shape)

        X = self.input_layer(X)

        # first down layer
        X1 = self.down1(X)
        print("X1 down:", X1.shape)

        # second down layer
        X2 = self.down2(X1)
        print("X2:", X2.shape)

        # third down layer
        X3 = self.down3(X2)
        print("X3:", X3.shape)

        # # fourth down layer
        X4 = self.down4(X3)
        print("X4 down4:", X4.shape)

        # # 5 down layer
        X5 = self.down5(X4)
        print("X5 down5:", X5.shape)

        # # 6 down layer
        X6 = self.down6(X5)
        print("X6 down6:", X6.shape)


        # # first up layer
        X5 = self.up1(X5, X6)
        print("X5 up1:", X5.shape)

        # # 2 up layer
        X4 = self.up2(X4, X5)
        print("X4 up2:", X4.shape)

        # # 3 up layer
        X3 = self.up3(X3, X4)
        print("X3 up3:", X3.shape)

        # # 4 up layer
        X2 = self.up4(X2, X3)
        print("X2 up4:", X2.shape)

        # # 5 up layer
        X1 = self.up5(X1, X2)
        print("X1 up5:", X1.shape)

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

        # polar = torch.polar(speech, phase)
        polar = speech * torch.cos(phase) + speech * torch.sin(phase) * 1j
        print("polar", polar.shape)
        speech_out = torch.istft(polar, self.fft_size, hop_length=self.hop_size, window=window, return_complex=False, onesided=True, center=True)
        # music_out = self.istft(music)

        print("speech_out:", speech_out.shape)
        print("x_in:", x_in.shape)
        # remove additional dimention
        speech_out = speech_out.squeeze(1)
        music_out = x_in - speech_out
        # music_out = music_out.squeeze(1)

        print("speech_out:", speech_out.shape)
        print("music_out:", music_out.shape)
        # add both sources to a tensor to return them
        T_data = torch.stack([speech_out, music_out], dim=1)

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
