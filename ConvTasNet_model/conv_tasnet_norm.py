from asteroid.models import ConvTasNet
import torch

class ConvTasNetNorm(ConvTasNet):
    def __init__(
            self,
            n_src,
            out_chan=None,
            n_blocks=8,
            n_repeats=3,
            bn_chan=128,
            hid_chan=512,
            skip_chan=128,
            conv_kernel_size=3,
            norm_type="gLN",
            mask_act="sigmoid",
            in_chan=None,
            causal=False,
            fb_name="free",
            kernel_size=16,
            n_filters=512,
            stride=8,
            encoder_activation=None,
            sample_rate=8000,
            **fb_kwargs,
        ):
        self.mean = None
        self.std = None
        super(ConvTasNetNorm, self).__init__(
            n_src,
            out_chan=out_chan,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            skip_chan=skip_chan,
            conv_kernel_size=conv_kernel_size,
            norm_type=norm_type,
            mask_act=mask_act,
            in_chan=in_chan,
            causal=causal,
            fb_name=fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            encoder_activation=encoder_activation,
            sample_rate=sample_rate,
            **fb_kwargs,
        )
    def forward_encoder(self, wav: torch.Tensor) -> torch.Tensor:
        # pre filter
        self.mean = torch.mean(wav)
        self.std = torch.std(wav)

        wav = (wav - self.mean) / (1e-5 + self.std)
        if torch.cuda.is_available():
            wav = wav.cuda()
        return super(ConvTasNetNorm, self).forward_encoder(wav)

    def forward_decoder(self, masked_tf_rep: torch.Tensor) -> torch.Tensor:
        waveforms = super(ConvTasNetNorm, self).forward_decoder(masked_tf_rep)
        waveforms = waveforms * self.std + self.mean

        return waveforms
        # return super(ConvTasNetNorm, self).forward_decoder(masked_tf_rep)

