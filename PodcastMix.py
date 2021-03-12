import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import os
import numpy as np
import random

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
    def __init__(self, csv_dir, sample_rate=44100, segment=3, return_id=False):
        self.csv_dir = csv_dir
        self.return_id = return_id
        self.speech_csv_path = os.path.join(self.csv_dir, 'speech.csv')
        self.music_csv_path = os.path.join(self.csv_dir, 'music.csv')
        self.segment = segment
        self.sample_rate = sample_rate
        # Open csv files
        self.df_speech = pd.read_csv(self.speech_csv_path, engine='python')
        self.df_music = pd.read_csv(self.music_csv_path, engine='python')
        self.seg_len = int(self.segment * self.sample_rate)

    def __len__(self):
        # for now, its a full permutation
        return 2000
        # return len(self.df_music) * len(self.df_speech)

    def compute_rand_offset_duration(self, audio_path):
        offset = duration = start = 0
        if self.segment is not None:
            info = torchaudio.info(audio_path)
            sr, channels, length = info.sample_rate, info.num_channels, info.num_frames
            duration = float(length / sr)
            if self.segment > duration:
                offset = 0
                num_frames = length
            else:
                # compute start in seconds
                start = random.uniform(0, duration - self.segment)
                offset = int(np.floor(start * sr))
                num_frames = int(np.floor(self.segment * sr))
        else:
            print('segment is empty, modify it in the config.yml file')
        return offset, num_frames

    def __getitem__(self, idx):
        speech_idx = idx // len(self.df_music)
        music_idx = idx % len(self.df_music)
        # Get the row in speech dataframe
        row_speech = self.df_speech.iloc[speech_idx]
        row_music = self.df_music.iloc[music_idx]
        sources_list = []

        # If there is a seg, start point is set randomly
        offset, num_frames = self.compute_rand_offset_duration(row_speech['speech_path'])
        # effects = [
        #     ['rate', str(self.sample_rate)],
        #     ['trim', '0', '3'],
        # ]
        # s_speech, _ = torchaudio.sox_effects.apply_effects_file(source_path, effects)
        source_path = row_speech["speech_path"]
        audio_signal, sr = torchaudio.load(filepath=source_path, frame_offset=offset, num_frames=num_frames)

        #### zero pad if the size is smaller than seq_duration
        seq_duration_samples = int(self.segment * sr)
        total_samples = audio_signal.shape[-1]
        # if seq_duration_samples>total_samples:
        #     audio_signal = torch.nn.ConstantPad2d((0,seq_duration_samples-total_samples,0,0),0)(audio_signal)

        # #### resample
        audio_signal = torchaudio.transforms.Resample(sr, self.sample_rate)(audio_signal)
        # s_speech = s_speech[0]
        # s_speech = s_speech.numpy()
        # # We want to cleanly separate Speech, so its the first source in the sources_list
        # #  s_speech, _ = torchaudio.load(source_path, frame_offset=offset, num_frames=duration, sr = self.sample_rate)
        # # Normalize speech
        # sources_list.append(s_speech)
        sources_list.append(audio_signal)


        sources_list.append(audio_signal)


        # now for music:


        # offset, num_frames = self.compute_rand_offset_duration(row_music['music_path'])
        # source_path = row_music["music_path"]
        # audio_signal, sr = torchaudio.load(filepath=source_path, frame_offset=offset, num_frames=num_frames)
        #### zero pad if the size is smaller than seq_duration
        # seq_duration_samples = int(self.segment * sr)
        # total_samples = audio_signal.shape[-1]
        # if seq_duration_samples>total_samples:
        #     audio_signal = torch.nn.ConstantPad2d((0,seq_duration_samples-total_samples,0,0),0)(audio_signal)


        # audio_signal = torchaudio.transforms.Resample(sr, self.sample_rate)(audio_signal)
        # stereo to mono:
        # audio_signal = audio_signal[0] + audio_signal[1]
        # sources_list.append(audio_signal)

        # compute the mixture
        mixture = sources_list[0] + 0.2 * sources_list[1]
        # Convert to torch tensor
        # mixture = torch.from_numpy(mixture)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        if not self.return_id:
            return mixture, sources
        return mixture, sources, [row_speech['speech_ID'], row_music['music_ID']]

    def get_infos(self):
        """Get dataset infos (for publishing models).
        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        return infos
