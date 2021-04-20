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
    def __init__(self, in_ch, out_ch, kernel_size, stride, output_padding, index):
        super(up, self).__init__()
        if index > 3:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, output_padding=output_padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        else:
            # 50% dropout for the first 3 layers
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, output_padding=output_padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            )

    def forward(self, x1, x2):
        x = self.deconv(x)
        x = torch.cat([x2, x1], dim=1)

        return x

class last_layer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, output_padding):
        super(last_layer, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, output_padding=output_padding),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.deconv(x)
        
        return x
        
