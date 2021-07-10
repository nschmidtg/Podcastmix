from PodcastMixSpec import PodcastMixSpec
import torch
import torchaudio
from pytorch_lightning import seed_everything
seed_everything(1, workers=True)

train_set = PodcastMixSpec(
    csv_dir='podcastmix/metadata/test',
    sample_rate=44100,
    segment=2,
    shuffle_tracks=True,
    multi_speakers=True,
    normalize=False,
    window_size=1024,
    fft_size=1024,
    hop_size=441,
)
a=train_set.__getitem__(0)
polar_mix = a[1][0][0] * torch.cos(a[1][0][1]) + a[1][0][0] * torch.sin(a[1][0][1]) * 1j
music = torch.istft(polar_mix, 1024, 441, window=torch.hamming_window(1024), return_complex=False, onesided=True, center=True)
torchaudio.save('podcast_fake.wav', music.unsqueeze(0), sample_rate=44100)