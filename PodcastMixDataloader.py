import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
import numpy as np
import random
from utils.resampler import Resampler

class PodcastMixDataloader(Dataset):
    dataset_name = "PodcastMix"

    def __init__(self, csv_dir, sample_rate=44100, original_sample_rate= 44100, segment=2,
                 shuffle_tracks=False, multi_speakers=False):
        self.csv_dir = csv_dir
        self.segment = segment
        # sample_rate of the original files
        self.original_sample_rate = original_sample_rate
        # destination sample_rate for resample
        self.sample_rate = sample_rate
        self.shuffle_tracks = shuffle_tracks
        self.multi_speakers = multi_speakers

        if not self.sample_rate == self.original_sample_rate:
            self.resampler = Resampler(
                input_sr=self.original_sample_rate,
                output_sr=self.sample_rate,
                dtype=torch.float32,
                filter='hann'
            )

        # declare dataframes
        self.speech_csv_path = os.path.join(self.csv_dir, 'speech.csv')
        self.music_csv_path = os.path.join(self.csv_dir, 'music.csv')
        self.df_speech = pd.read_csv(self.speech_csv_path, engine='python')

        # dictionary of speakers
        self.speakers_dict = {}
        for speaker_id in self.df_speech.speaker_id.unique():
            self.speakers_dict[speaker_id] = self.df_speech.loc[
                self.df_speech['speaker_id'] == speaker_id
            ]
        self.df_music = pd.read_csv(self.music_csv_path, engine='python')

        # initialize indexes
        self.speech_inxs = list(range(len(self.df_speech)))
        self.music_inxs = list(range(len(self.df_music)))

        # declare the resolution of the reduction factor.
        # this will create N different gain values max
        # 1/denominator_gain to multiply the music gain
        self.denominator_gain = 20
        self.gain_ramp = np.array(range(1, self.denominator_gain, 1))/self.denominator_gain

        # shuffle the static random gain to use it in testing
        np.random.shuffle(self.gain_ramp)

        # use soundfile as backend
        torchaudio.set_audio_backend(backend='soundfile')

    def __len__(self):
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
        length = int(row['length'])
        audio_signal = torch.zeros(self.segment * self.original_sample_rate)
        # iterate until the segment is not silence
        audio_signal = self.load_mono_random_segment(audio_signal, length, row['music_path'], self.segment * self.original_sample_rate)

        # zero pad if the size is smaller than seq_duration
        seq_duration_samples = int(self.segment * self.original_sample_rate)
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
        """
        concatenates random speech files from the same speaker as speech_idx until
        obtaining a buffer with a length of at least the lenght of the
        input segment.
        If multispeaker is used, a single audio file from a different
        speaker is overlapped in a random position of the buffer, to emulate
        speakers interruptions once every 10 items.
        The buffer is shifted in a random position to prevent always getting
        buffers that starts with the beginning of a speech.
        Returns the shifted buffer with a length equal to segment.
        """
        speaker_csv_id = self.df_speech.iloc[speech_idx].speaker_id
        array_size = self.original_sample_rate * self.segment
        speech_mix = torch.zeros(0)
        speech_counter = 0
        while speech_counter < array_size:
            # file is shorter than segment, concatenate with more until
            # is at least the same length
            row_speech = self.speakers_dict[speaker_csv_id].sample()
            audio_path = row_speech['speech_path'].values[0]
            speech_signal, _ = torchaudio.load(
                audio_path
            )
            # add the speech to the buffer
            speech_mix = torch.cat((speech_mix, speech_signal[0]))
            speech_counter += speech_signal.shape[-1]

        # we have a segment of at least self.segment length speech audio
        # from the same speaker
        if self.multi_speakers and speech_idx % 10 == 0:
            # every 10 iterations overlap another speaker
            list_of_speakers = list(self.speakers_dict.keys())
            list_of_speakers.remove(speaker_csv_id)
            non_speaker_id = random.sample(list_of_speakers, 1)[0]
            row_speech = self.speakers_dict[non_speaker_id].sample()
            audio_path = row_speech['speech_path'].values[0]
            other_speech_signal, _ = torchaudio.load(
                audio_path
            )

            other_speech_signal_length = other_speech_signal.shape[-1]
            if len(speech_mix) < other_speech_signal.shape[-1]:
                # the second speaker is longer than the original one
                other_speech_signal_length = len(speech_mix)
            offset = random.randint(0, len(speech_mix) - other_speech_signal_length)
            speech_mix[offset:offset + other_speech_signal_length] += other_speech_signal[0][:other_speech_signal_length]
            speech_mix = speech_mix / 2

        # we have a segment with the two speakers, the second in a random start.
        # now we randomly shift the array to pick the start
        offset = random.randint(0, array_size)
        zeros_aux = torch.zeros(len(speech_mix))
        aux = speech_mix[:offset]
        
        zeros_aux[:len(speech_mix) - offset] = speech_mix[offset:len(speech_mix)]
        zeros_aux[len(speech_mix) - offset:] = aux
            
        return zeros_aux[:array_size]

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

        if not self.sample_rate == self.original_sample_rate:
            speech_signal = self.resampler.forward(speech_signal)
            music_signal = self.resampler.forward(music_signal)

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
        
        # append sources:
        sources_list.append(speech_signal)
        sources_list.append(music_signal)

        # compute the mixture as the avg of both sources
        mixture = 0.5 * (sources_list[0] + sources_list[1])
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
