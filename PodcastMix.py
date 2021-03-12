import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
import pandas as pd
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
        # Get rid of the utterances too short
        if self.segment is not None:
            max_len = len(self.df_music)
            self.seg_len = int(self.segment * self.sample_rate)
            # Ignore the file shorter than the desired_length
            self.df_music = self.df_music[self.df_music["length"] >= self.seg_len]
            print(
                f"Music: Drop {max_len - len(self.df_music)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )

            max_len = len(self.df_speech)
            self.seg_len = int(self.segment * self.sample_rate)
            # Ignore the file shorter than the desired_length
            self.df_speech = self.df_speech[self.df_speech["length"] >= self.seg_len]
            print(
                f"Speech: Drop {max_len - len(self.df_speech)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None

    def __len__(self):
        # for now, its a full permutation
        return 2000
        # return len(self.df_music) * len(self.df_speech)

    def compute_rand_offset_duration(self, row):
        offset = duration = 0
        if self.segment is not None:
            # ARREGLAR! offset = float(random.randint(0, row["length"] - self.seg_len) / self.sample_rate)
            offset = 0
            duration = self.seg_len
        else:
            offset = 0
            duration = None
        return offset, duration

    def __getitem__(self, idx):
        #### Marius code:
        # #### compute start and end frames to read from the disk
        # si, ei = torchaudio.info(track.audio_path)
        # sample_rate, channels, length = si.rate, si.channels, si.length
        # duration = length / sample_rate
        # if self.seq_duration>duration:
        #     offset = 0
        #     num_frames = length
        # else:
        #     if self.random_start:
        #         #### we skip a number of seconds at the beginning of file 
        #         start = random.uniform(0, duration - self.seq_duration)
        #     else:
        #         start = 0.
        #     #### seconds to audio frames
        #     offset = int(np.floor(start * sample_rate))
        #     num_frames = int(np.floor(self.seq_duration * sample_rate))


        # #### get audio frames corresponding to offset and num_frames from the disk
        # audio_signal, sample_rate = torchaudio.load(filepath=track.audio_path, offset=offset,num_frames=num_frames)

        # #### zero pad if the size is smaller than seq_duration
        # seq_duration_samples = int(self.seq_duration * sample_rate)
        # total_samples = audio_signal.shape[-1]
        # if seq_duration_samples>total_samples:
        #     audio_signal = torch.nn.ConstantPad2d((0,seq_duration_samples-total_samples,0,0),0)(audio_signal)

        # #### resample
        # audio_signal = torchaudio.transforms.Resample(sample_rate, self.resample)(audio_signal)




        speech_idx = idx // len(self.df_music)
        music_idx = idx % len(self.df_music)
        # Get the row in speech dataframe
        row_speech = self.df_speech.iloc[speech_idx]
        row_music = self.df_music.iloc[music_idx]
        sources_list = []

        # If there is a seg, start point is set randomly
        offset, duration = self.compute_rand_offset_duration(row_speech)
        effects = [
            ['rate', str(self.sample_rate)],
            ['trim', '0', '3'],
        ]
        source_path = row_speech["speech_path"]
        s_speech, _ = torchaudio.sox_effects.apply_effects_file(source_path, effects)
        s_speech = s_speech[0]
        s_speech = s_speech.numpy()
        # We want to cleanly separate Speech, so its the first source in the sources_list
        #  s_speech, _ = torchaudio.load(source_path, frame_offset=offset, num_frames=duration, sr = self.sample_rate)
        # Normalize speech
        s_speech = s_speech / max(s_speech)
        sources_list.append(s_speech)

        # now for music:
        offset, duration = self.compute_rand_offset_duration(row_music)
        effects = [
            ['rate', str(self.sample_rate)],
            ['trim', '0', '3'],
        ]
        source_path = row_music["music_path"]
        offset, duration = self.compute_rand_offset_duration(row_music)
        s_music, _ = torchaudio.sox_effects.apply_effects_file(source_path, effects)
        s_music = s_music[0]
        # normalize:
        s_music = s_music / max(s_music)
        s_music = s_music.numpy()
        sources_list.append(s_music)

        # compute the mixture
        mixture = s_music * 0.2 + s_speech
        # Convert to torch tensor
        mixture = torch.from_numpy(mixture)
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
