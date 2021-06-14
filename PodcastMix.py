import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
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

    def __init__(self, csv_dir, sample_rate=44100, segment=2,
                 shuffle_tracks=False, multi_speakers=False):
        self.csv_dir = csv_dir
        self.speech_csv_path = os.path.join(self.csv_dir, 'speech.csv')
        self.music_csv_path = os.path.join(self.csv_dir, 'music.csv')
        self.segment = segment
        self.segment_total = 12
        self.sample_rate = sample_rate
        # self.solo_music_ratio = solo_music_ratio
        self.shuffle_tracks = shuffle_tracks
        self.multi_speakers = multi_speakers
        # Open csv files
        self.df_speech = pd.read_csv(self.speech_csv_path, engine='python')
        self.df_music = pd.read_csv(self.music_csv_path, engine='python')
        # self.seg_len = int(self.segment * self.sample_rate)
        # initialize indexes
        self.speech_inxs = list(range(len(self.df_speech)))
        self.music_inxs = list(range(len(self.df_music)))
        self.denominator_gain = 10
        self.gain_ramp = np.array(range(1, self.denominator_gain, 1))/self.denominator_gain
        np.random.shuffle(self.gain_ramp)
        torchaudio.set_audio_backend(backend='soundfile')

    def __len__(self):
        # for now, its a full permutation
        # return 50
        return min([len(self.df_speech), len(self.df_music)])


    def compute_rand_offset_duration(self, original_sr, original_num_frames, segment):
        """ Computes a random offset and the number of frames to read the audio_path
        in order to get a subsection of length equal to self.segment. If the number 
        of frames in the segment is bigger than the original_num_frames, it returns
        0 for the offset and the number of frames contained in the audio.

        Parameters:
        - audio_path (str) : path to the audio file
        - original_sr (int) : sample rate of the audio file
        - original_num_frames (int) : number of frames of the full audio file

        Returns (tuple):
        - offset (int) : the computed random offset in frames where the audio should be 
        loaded
        - segment_frames (int) : the number of frames contained in the segment. If number
        of frames contained in the audio is smaller than the number of frames contained
        in the segment at the self.sample_rate sample rate, the offset is 0 and the 
        segment_frames will be equal to the length of the file
        """
        offset = 0
        segment_frames = int(segment * original_sr)
        if segment_frames > original_num_frames:
            segment_frames = original_num_frames
        else:
            offset = int(random.uniform(0, original_num_frames - segment_frames))

        return offset, segment_frames

    def load_mono_non_silent_random_segment(self, row, music_or_speech):
        """ Randomly selects a non_silent part of the audio given by audio_path

        Parameters:
        - audio_path (str) : path to the audio file

        Returns:
        - audio_signal (torchaudio) : waveform of the
        """
        # info = torchaudio.info(audio_path)
        # sr, length = info.sample_rate, info.num_frames
        sr = 44100
        length = int(row['length'])
        audio_signal = torch.tensor([0.])
        offset=duration=0
        # iterate until the segment is not silence
        while torch.sum(torch.abs(audio_signal)) == 0:
            # If there is a seg, start point is set randomly
            offset, duration = self.compute_rand_offset_duration(
                sr,
                length,
                self.segment_total
            )
            # load the audio with the computed offsets
            audio_signal, sr = torchaudio.load(
                row[music_or_speech + '_path'],
                frame_offset=offset,
                num_frames=duration
            )
        # convert to mono
        if len(audio_signal) == 2:
            audio_signal = torch.mean(audio_signal, dim=0).unsqueeze(0)
        # resample if sr is different than the specified in dataloader
        if not sr == self.sample_rate:
            audio_signal = torchaudio.transforms.Resample(orig_freq = sr, new_freq = self.sample_rate)(audio_signal)
        # zero pad if the size is smaller than seq_duration
        seq_duration_samples = int(self.segment_total * self.sample_rate)
        total_samples = audio_signal.shape[-1]
        if seq_duration_samples > total_samples:
            padding_offset = random.randint(0, seq_duration_samples - total_samples)
            audio_signal = torch.nn.ConstantPad1d(
                (
                    padding_offset,
                    seq_duration_samples - total_samples - padding_offset
                ),
                0
            )(audio_signal)

        return audio_signal

    def rms(self, audio):
        """ computes the RMS of an audio signal
        """

        return torch.sqrt(torch.mean(audio ** 2))

    def load_speechs(self, speech_idx):
        """ Loads the speaker mix. It could be a single speaker if
        self.multi_speaker=False, or a random mix between 1 to 4
        speakers
        """
        speech_mix = []
        # number_of_speakers = random.randint(1, 4) if self.multi_speakers else 1
        speaker_csv_id = self.df_speech.iloc[speech_idx].speaker_id
        # filter the speechs by speaker id
        speaker_dict = self.df_speech.loc[
            self.df_speech['speaker_id'] == speaker_csv_id
        ]
        speech_counter = 0
        speech_signal = torch.tensor([0.])
        sr = 44100
        while speech_counter < (self.segment_total * self.sample_rate):
            row_speech = speaker_dict.sample()
            audio_length = int(row_speech['length'])
            audio_path = row_speech['speech_path'].values[0]
            while torch.sum(torch.abs(speech_signal)) == 0:
                # If there is a seg, start point is set randomly
                offset, duration = self.compute_rand_offset_duration(
                    sr,
                    audio_length,
                    (self.segment_total * self.sample_rate) - speech_counter
                )
                # load the audio with the computed offsets
                speech_signal, sr = torchaudio.load(
                    audio_path,
                    frame_offset=offset,
                    num_frames=duration
                )
            speech_counter += duration

            # row_speech = self.df_speech.iloc[speech_idx]
            # speech_signal = self.load_mono_non_silent_random_segment(row_speech, 'speech')
            speech_mix.append(speech_signal)
            # re-initialize signal
            speech_signal = torch.tensor([0.])
        speech_mix = torch.cat((speech_mix), 1)
        #speech_mix = speech_mix.squeeze(0)
        if self.multi_speakers and speech_idx % 10 == 0:
            # every 10 iterations overlap another speaker
            non_speaker_dict = self.df_speech.loc[
                self.df_speech['speaker_id'] != speaker_csv_id
            ]
            speech_signal = torch.tensor([0.])
            row_speech = non_speaker_dict.sample()
            audio_length = int(row_speech['length'])
            audio_path = row_speech['speech_path'].values[0]
            while torch.sum(torch.abs(speech_signal)) == 0:
                # If there is a seg, start point is set randomly
                offset, duration = self.compute_rand_offset_duration(
                    sr,
                    audio_length,
                    self.segment
                )
                # load the audio with the computed offsets
                speech_signal, sr = torchaudio.load(
                    audio_path,
                    frame_offset=offset,
                    num_frames=duration
                )
            offset, duration = self.compute_rand_offset_duration(
                sr,
                audio_length,
                self.segment_total
            )
            speech_mix[offset:offset + duration] += speech_signal[offset:offset + duration]
        #speech_mix = torch.Tensor(speech_mix)
        # speech_mix = torch.mean(speech_mix, dim=0) if self.multi_speakers else speech_mix
        speech_mix = speech_mix.squeeze(1)
        return speech_mix

    def normalize_audio(self, track):
        """
        Receives mono audio and the normalize it
        """
        return (track - torch.mean(track)) / torch.std(track.std)

    def __getitem__(self, idx):
        if(idx == 0 and self.shuffle_tracks):
            # shuffle on first epochs of training and validation. Not testing
            random.shuffle(self.music_inxs)
            random.shuffle(self.speech_inxs)
        # get corresponding index from the list
        music_idx = self.music_inxs[idx]
        speech_idx = self.speech_inxs[idx]
        # Get the row in speech dataframe
        row_music = self.df_music.iloc[music_idx]
        sources_list = []

        # We want to cleanly separate Speech, so its the first source
        # in the sources_list
        speech_signal = self.load_speechs(speech_idx)
        music_signal = self.load_mono_non_silent_random_segment(row_music, 'music')
        speech_cropped = torch.zeros(1)
        music_cropped = torch.zeros(1)
        while torch.sum(torch.abs(speech_cropped)) == 0 or torch.sum(torch.abs(music_cropped)) == 0:
            offset_truncate = int(random.uniform(0, self.segment_total * self.sample_rate - self.segment * self.sample_rate - 1))
            speech_cropped = speech_signal[..., offset_truncate:offset_truncate + (self.segment * self.sample_rate)]
            music_cropped = music_signal[..., offset_truncate:offset_truncate + (self.segment * self.sample_rate)]
        speech_signal = speech_cropped
        music_signal = music_cropped
        speech_signal = self.normalize_audio(speech_signal)
        sources_list.append(speech_signal)
        # gain based on RMS in order to have RMS(speech_signal) >= RMS(music_singal)
        reduction_factor = self.rms(speech_signal) / self.rms(music_signal)
        # now we know that rms(r * music_signal) == rms(speech_signal)
        if self.shuffle_tracks:
            # random gain for training and validation
            music_gain = random.uniform(1/self.denominator_gain, 1) * reduction_factor
        else:
            # fixed gain for testing
            music_gain = self.gain_ramp[idx % len(self.gain_ramp)] * reduction_factor

        # multiply the music by the gain factor and add to the sources_list
        music_signal = music_gain * music_signal
        music_signal = self.normalize_audio(music_signal)
        sources_list.append(music_signal)
        # compute the mixture
        mixture = 0.5 * (sources_list[0] + sources_list[1])
        mixture = self.normalize_audio(mixture)
        mixture = torch.squeeze(mixture)

        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)

        return mixture, sources

    def get_infos(self):
        """Get dataset infos (for publishing models).
        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        return infos
