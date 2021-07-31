from asteroid.models import ConvTasNet
import torch

class ConvTasNetNorm(ConvTasNet):
    self.mean = None
    self.std = None
    super().__init__(
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
    )

    def forward_encoder(self, wav: torch.Tensor) -> torch.Tensor:
        # pre filter
        self.mean = torch.mean(wav)
        self.std = torch.std(wav)
        wav = (wav - self.mean) / (1e-5 + self.std)
        super().forward_encoder(self, wav)

    def forward_decoder(self, masked_tf_rep: torch.Tensor) -> torch.Tensor:
        waveforms = super().forward_decoder(self, masked_tf_rep)
        waveforms = waveforms * self.std + self.mean
        return waveforms
