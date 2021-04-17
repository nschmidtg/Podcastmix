# subparts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
class down(nn.Module):
    '''(conv => BN => LeackyReLU)'''
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class up(nn.Module):
    '''
        (concatenation => deconv => BN => ReLU => Dropout )
        as proposed in SINGING VOICE SEPARATION WITH DEEP
        U-NET CONVOLUTIONAL NETWORK
    '''
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(up, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(17, out_ch, kernel_size, stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

    def forward(self, x1, x2):
        # input is CHW. Taken from 
        # https://github.com/Steve-Tod/Audio-source-separation-with-Unet/blob/master/models/unet/unet_parts.py
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.deconv(x)

        return x
