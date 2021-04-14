import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import os
import numpy as np
import random
import librosa


class PodcastMix(Dataset):
    """Dataset class for PodcastMix source separation tasks.
    This dataset created Podcasts-like mixes on the fly consisting in
    2 sources: background music and foreground speech
    Args:
        csv_dir (str): The path to the metadata files: speech.csv and music.csv
        sample_rate (int) : The sample rate of the sources and mixtures.
        segment (int) : The desired sources and mixtures length in s.
    References
        [1] "Jamendo....
        [2] "VCTK...
    """

    dataset_name = "PodcastMix"

    def __init__(self, csv_dir, sample_rate=44100, segment=3, return_id=False, 
                 shuffle_tracks=False):
        self.csv_dir = csv_dir
        self.return_id = return_id
        self.speech_csv_path = os.path.join(self.csv_dir, 'speech.csv')
        self.music_csv_path = os.path.join(self.csv_dir, 'music.csv')
        self.segment = segment
        self.sample_rate = sample_rate
        self.shuffle_tracks = shuffle_tracks
        # Open csv files
        self.df_speech = pd.read_csv(self.speech_csv_path, engine='python')
        self.df_music = pd.read_csv(self.music_csv_path, engine='python')
        self.seg_len = int(self.segment * self.sample_rate)
        # initialize indexes
        self.speech_inxs = list(range(len(self.df_speech)))
        self.music_inxs = list(range(len(self.df_music)))
        np.random.seed(1)
        random.seed(1)
        self.gain_ramp = np.array(range(1, 100, 1))/100
        np.random.shuffle(self.gain_ramp)

    def __len__(self):
        # for now, its a full permutation
        return min([len(self.df_speech), len(self.df_music)])

    def compute_rand_offset_duration(self, audio_path):
        offset = duration = start = 0
        if self.segment is not None:
            info = torchaudio.info(audio_path)
            sr, length = info.sample_rate, info.num_frames
            duration = length
            segment_frames = int(self.segment * sr)
            if segment_frames > duration:
                offset = 0
                num_frames = segment_frames
            else:
                # compute start in seconds
                start = int(random.uniform(0, duration - segment_frames))
                offset = start
                num_frames = segment_frames
        else:
            print('segment is empty, modify it in the config.yml file')
        return offset, num_frames

    def __getitem__(self, idx):
        if(idx == 0 and self.shuffle_tracks):
            # shuffle on first epochs of training and validation. Not testing
            random.shuffle(self.music_inxs)
            random.shuffle(self.speech_inxs)
        # get corresponding index from the list
        speech_idx = self.speech_inxs[idx]
        music_idx = self.music_inxs[idx]
        # Get the row in speech dataframe
        row_speech = self.df_speech.iloc[speech_idx]
        row_music = self.df_music.iloc[music_idx]
        sources_list = []

        audio_signal = torch.tensor([0.])
        sr = 0
        while torch.sum(torch.abs(audio_signal)) == 0:
            # If there is a seg, start point is set randomly
            offset, duration = self.compute_rand_offset_duration(
                row_speech['speech_path']
            )
            # We want to cleanly separate Speech, so its the first source
            # in the sources_list
            source_path = row_speech["speech_path"]
            audio_signal, sr = torchaudio.load(
                source_path,
                frame_offset=offset,
                num_frames=duration,
                normalize=True
            )
            # resample if sr is different than the specified in dataloader
            if not sr == self.sample_rate:
                audio_signal = torchaudio.transforms.Resample(orig_freq = sr, new_freq = self.sample_rate)(audio_signal)
            # zero pad if the size is smaller than seq_duration
            seq_duration_samples = int(self.segment * sr)
            total_samples = audio_signal.shape[-1]
            if seq_duration_samples > total_samples:
                audio_signal = torch.nn.ConstantPad2d(
                    (
                        0,
                        seq_duration_samples - total_samples,
                        0,
                        0),
                    0
                )(audio_signal)
        sources_list.append(audio_signal)

        # now for music:
        audio_signal = torch.tensor([0.])
        sr = 0
        while torch.sum(torch.abs(audio_signal)) == 0:
            # If there is a seg, start point is set randomly
            offset, duration = self.compute_rand_offset_duration(
                row_music['music_path']
            )
            source_path = row_music["music_path"]
            audio_signal, sr = torchaudio.load(
                source_path,
                frame_offset=offset,
                num_frames=duration,
                normalize=True
            )
            # resample if sr is different than the specified in dataloader
            if not sr == self.sample_rate:
                audio_signal = torchaudio.transforms.Resample(orig_freq = sr, new_freq = self.sample_rate)(audio_signal)
            # zero pad if the size is smaller than seq_duration
            seq_duration_samples = int(self.segment * sr)
            total_samples = audio_signal.shape[-1]
            if seq_duration_samples > total_samples:
                audio_signal = torch.nn.ConstantPad2d(
                    (
                        0,
                        seq_duration_samples - total_samples,
                        0,
                        0),
                    0
                )(audio_signal)

        if len(audio_signal) == 2:
            audio_signal = audio_signal[0] + audio_signal[1]

        if self.shuffle_tracks:
            # random gain for training and validation
            music_gain = random.uniform(1e-3, 1)
        else:
            # fixed gain for testing
            music_gain = self.gain_ramp[idx % len(self.gain_ramp)]

        # multiply the music by the gain factor and add to the sources_list
        sources_list.append(music_gain * audio_signal)
        # compute the mixture
        mixture = sources_list[0] + sources_list[1]
        mixture = torch.squeeze(mixture)

        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        # print(sources[0].shape)
        # print(sources[1].shape)
        # print(mixture.shape)
        if not self.return_id:
            return mixture, sources
        return mixture, sources, [
            row_speech['speech_ID'],
            row_music['music_ID']
        ]

    def get_infos(self):
        """Get dataset infos (for publishing models).
        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        return infos
