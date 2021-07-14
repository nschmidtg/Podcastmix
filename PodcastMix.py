import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
import numpy as np
import random
from resampler import Resampler

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
                 shuffle_tracks=False, multi_speakers=False, normalize=True):
        self.csv_dir = csv_dir
        self.speech_csv_path = os.path.join(self.csv_dir, 'speech.csv')
        self.music_csv_path = os.path.join(self.csv_dir, 'music.csv')
        self.segment = segment
        self.segment_total = 12
        self.sample_rate = sample_rate
        # print("sample_rate", self.sample_rate)
        self.normalize = normalize
        # resampler
        self.resampler = Resampler(input_sr = 44100, output_sr = self.sample_rate, dtype=torch.float32, filter='hann', num_zeros=6)
        # self.solo_music_ratio = solo_music_ratio
        self.shuffle_tracks = shuffle_tracks
        self.multi_speakers = multi_speakers
        # Open csv files
        self.df_speech = pd.read_csv(self.speech_csv_path, engine='python')
        self.speakers_dict = {}
        for speaker_id in self.df_speech.speaker_id.unique():
            self.speakers_dict[speaker_id] = self.df_speech.loc[
                self.df_speech['speaker_id'] == speaker_id
            ]
        self.df_music = pd.read_csv(self.music_csv_path, engine='python')
        # self.seg_len = int(self.segment * self.sample_rate)
        # initialize indexes
        self.speech_inxs = list(range(len(self.df_speech)))
        self.music_inxs = list(range(len(self.df_music)))
        self.denominator_gain = 10000
        self.gain_ramp = np.array(range(1, self.denominator_gain, 1))/self.denominator_gain
        np.random.shuffle(self.gain_ramp)
        torchaudio.set_audio_backend(backend='soundfile')

    def __len__(self):
        # for now, its a full permutation
        return 500
        return min([len(self.df_speech), len(self.df_music)])

    def compute_rand_offset_duration(self, original_num_frames, segment_frames):
        """ Computes a random offset and the number of frames to read from a file
        in order to get a rantom subsection of length equal to segment. If segment_frames
         is bigger than the original_num_frames, it returns
        0 for the offset and the number of frames contained in the audio. so its shorter
        than the desired segment.

        Parameters:
        - original_num_frames (int) : number of frames of the full audio file
        - segment_frames (int): number of frames of the desired segment

        Returns (tuple):
        - offset (int) : the computed random offset in frames where the audio should be 
        loaded
        - segment_frames (int) : the number of frames contained in the segment.
        """
        offset = 0
        if segment_frames > original_num_frames:
            # segment: |....|
            # audio:   |abc|
            # the file is shorter than the desired segment: |abc|
            segment_frames = original_num_frames
        else:
            # segment: |....|
            # audio:   |abcdefg|
            # the file is longer than the desired segment: |cdef|
            offset = int(random.uniform(0, original_num_frames - segment_frames))

        return offset, segment_frames
    
    def load_mono_random_segment(self, audio_signal, audio_length, audio_path, max_segment):
        while audio_length - torch.count_nonzero(audio_signal) == audio_length:
            # If there is a seg, start point is set randomly
            offset, duration = self.compute_rand_offset_duration(
                audio_length,
                max_segment
            )
            # load the audio with the computed offsets
            audio_signal, _ = torchaudio.load(
                audio_path,
                frame_offset=offset,
                num_frames=duration
            )
        # convert to mono
        if len(audio_signal) == 2:
            audio_signal = torch.mean(audio_signal, dim=0).unsqueeze(0)
        return audio_signal

    def load_non_silent_random_music(self, row):
        """ Randomly selects a non_silent part of the audio given by audio_path

        Parameters:
        - audio_path (str) : path to the audio file

        Returns:
        - audio_signal (torchaudio) : waveform of the
        """
        # info = torchaudio.info(audio_path)
        # music sample_rate
        sr = 44100
        length = int(row['length'])
        audio_signal = torch.zeros(self.segment_total * sr)
        # iterate until the segment is not silence
        audio_signal = self.load_mono_random_segment(audio_signal, length, row['music_path'], self.segment_total * sr)
        # print("music raw", audio_signal)
        # resample if sr is different than the specified in dataloader
        # print("audio signal", audio_signal.shape)
        # if not sr == self.sample_rate:
            # print("audiosignallll", audio_signal.shape)
            # audio_signal = self.resampler.forward(audio_signal)

        # zero pad if the size is smaller than seq_duration
        seq_duration_samples = int(self.segment_total * sr)
        total_samples = audio_signal.shape[-1]
        if seq_duration_samples > total_samples:
            # add zeros at beginning and at with random offset
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
        speech_counter = 0
        sr = 44100
        speech_signal = torch.zeros(1)
        # original speech sample_rate
        while speech_counter < (self.segment_total * sr): # have not resampled yet
            row_speech = self.speakers_dict[speaker_csv_id].sample()
            audio_length = int(row_speech['length'])
            audio_path = row_speech['speech_path'].values[0]
            speech_signal = self.load_mono_random_segment(speech_signal, audio_length, audio_path, (self.segment_total * sr) - speech_counter)
            speech_mix.append(speech_signal)
            speech_counter += speech_signal.shape[-1]
            # re-initialize empty audio signal
            speech_signal = torch.zeros(self.segment_total * sr)
        speech_mix = torch.cat((speech_mix), 1)
        #speech_mix = speech_mix.squeeze(0)
        if self.multi_speakers and speech_idx % 10 == 0:
            # every 10 iterations overlap another speaker
            
            speech_signal = torch.zeros(self.segment * sr)
            list_of_speakers = list(self.speakers_dict.keys())
            list_of_speakers.remove(speaker_csv_id)
            non_speaker_id = random.sample(list_of_speakers, 1)[0]
            row_speech = self.speakers_dict[non_speaker_id].sample()
            audio_length = int(row_speech['length'])
            audio_path = row_speech['speech_path'].values[0]
            speech_signal = self.load_mono_random_segment(speech_signal, audio_length, audio_path, self.segment * sr)
            offset = random.randint(0, self.segment_total * sr - speech_signal.shape[-1])
            speech_mix[..., offset:offset + speech_signal.shape[-1]] += speech_signal[...,:]
        speech_mix = speech_mix.squeeze(1)
        # resample if sr is different than the specified in dataloader
        # if not sr == self.sample_rate:
        #     speech_mix = self.resampler.forward(speech_mix)
        return speech_mix

    def normalize_audio(self, sources, mixture):
        """
        Receives mono audio and the normalize it
        """
        ref = mixture
        mixture = (mixture - torch.mean(ref)) / torch.std(ref)
        sources = (sources - torch.mean(ref)) / torch.std(ref)
        return sources, mixture

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
        music_signal = self.load_non_silent_random_music(row_music)
        
        target_resampled_number_samples = self.segment * self.sample_rate
        speech_cropped = torch.zeros(target_resampled_number_samples)
        music_cropped = torch.zeros(target_resampled_number_samples)
        while (torch.count_nonzero(speech_cropped)) == 0 or (torch.count_nonzero(music_cropped) == 0):
            offset_truncate = int(random.uniform(0, music_signal.shape[-1] - target_resampled_number_samples - 1))
            speech_cropped = speech_signal[..., offset_truncate:offset_truncate + (target_resampled_number_samples)]
            music_cropped = music_signal[..., offset_truncate:offset_truncate + (target_resampled_number_samples)]
        speech_signal = speech_cropped
        music_signal = music_cropped
        if not self.sample_rate == 44100:
            speech_signal = self.resampler.forward(speech_signal)
            music_signal = self.resampler.forward(music_signal)
        # append speech
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
        # append music
        sources_list.append(music_signal)
        # compute the mixture as the avg of both sources
        mixture = 0.5 * (sources_list[0] + sources_list[1])
        mixture = torch.squeeze(mixture)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        if self.normalize:
            sources, mixture = self.normalize_audio(sources, mixture)

        return mixture, sources

    def get_infos(self):
        """Get dataset infos (for publishing models).
        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        return infos
