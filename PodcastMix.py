import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import soundfile as sf
import os
import numpy as np
import random

class PodcastMix(Dataset):
    """Dataset class for PodcastMix source separation tasks.
    Args:
        csv_dir (str): The path to the metadata file.
        task (str): One of ``'linear_mono'``, ``'linear_stereo'``, 
            ``'sidechain_mono'`` or ``'sidechain_stereo'`` :
            * ``'linear_mono'`` for linear_mono mix
            * ``'linear_stereo'`` for linear_stereo mix
        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The number of sources in the mixture.
        segment (int) : The desired sources and mixtures length in s.
    References
        [1] "LibriMix: An Open-Source Dataset for Generalizable Speech Separation",
        Cosentino et al. 2020.
        [2] "MUSDB18 - a corpus for music separation",
        Zafar et al. 2018.
    """

    dataset_name = "PodcastMix"

    def __init__(self, csv_dir, task="linear_mono", sample_rate=44100, n_src=2, segment=3):
        self.csv_dir = csv_dir
        self.task = task
        # Get the csv corresponding to the task
        md_file = [f for f in os.listdir(csv_dir) if task in f][0]
        self.csv_path = os.path.join(self.csv_dir, md_file)
        self.segment = segment
        self.sample_rate = sample_rate
        # Open csv file
        self.df = pd.read_csv(self.csv_path, engine='python')
        # Get rid of the utterances too short
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df["length"] >= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None
        self.n_src = n_src

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]
        # Get mixture path
        self.mixture_path = row["mixture_path"]
        sources_list = []
        # If there is a seg start point is set randomly
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None
        # # If task is enh_both then the source is the clean mixture
        # if "enh_both" in self.task:
        #     mix_clean_path = self.df_clean.iloc[idx]["mixture_path"]
        #     s, _ = sf.read(mix_clean_path, dtype="float32", start=start, stop=stop)
        #     sources_list.append(s)

        # else:
            # Read sources
            # for i in range(self.n_src):
        source_path = row["track_path"]
        s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
        sources_list.append(s)

        source_path = row["speech_path"]
        s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
        sources_list.append(s)
        # Read the mixture
        mixture, _ = sf.read(self.mixture_path, dtype="float32", start=start, stop=stop)
        # Convert to torch tensor
        mixture = torch.from_numpy(mixture)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        return mixture, sources

    @classmethod
    def loaders_from_mini(cls, batch_size=4, **kwargs):
        """Downloads MiniLibriMix and returns train and validation DataLoader.
        Args:
            batch_size (int): Batch size of the Dataloader. Only DataLoader param.
                To have more control on Dataloader, call `mini_from_download` and
                instantiate the DatalLoader.
            **kwargs: keyword arguments to pass the `LibriMix`, see `__init__`.
                The kwargs will be fed to both the training set and validation
                set.
        Returns:
            train_loader, val_loader: training and validation DataLoader out of
            `LibriMix` Dataset.
        Examples
            >>> from asteroid.data import LibriMix
            >>> train_loader, val_loader = LibriMix.loaders_from_mini(
            >>>     task='sep_clean', batch_size=4
            >>> )
        """
        train_set, val_set = cls.mini_from_download(**kwargs)
        train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=True)
        return train_loader, val_loader

    @classmethod
    def mini_from_download(cls, **kwargs):
        """Downloads MiniLibriMix and returns train and validation Dataset.
        If you want to instantiate the Dataset by yourself, call
        `mini_download` that returns the path to the path to the metadata files.
        Args:
            **kwargs: keyword arguments to pass the `LibriMix`, see `__init__`.
                The kwargs will be fed to both the training set and validation
                set
        Returns:
            train_set, val_set: training and validation instances of
            `LibriMix` (data.Dataset).
        Examples
            >>> from asteroid.data import LibriMix
            >>> train_set, val_set = LibriMix.mini_from_download(task='sep_clean')
        """
        # kwargs checks
        assert "csv_dir" not in kwargs, "Cannot specify csv_dir when downloading."
        # assert kwargs.get("task", "sep_clean") in [
        #     "sep_clean",
        #     "sep_noisy",
        # ], "Only clean and noisy separation are supported in MiniLibriMix."
        assert (
            kwargs.get("sample_rate", 44100) == 44100
        ), "Only 44100kHz sample rate is supported in MiniLibriMix."
        # Download LibriMix in current directory
        meta_path = 'augmented_dataset/metadata'
        # Create dataset instances
        train_set = cls(os.path.join(meta_path, "train"), sample_rate=44100, **kwargs)
        val_set = cls(os.path.join(meta_path, "val"), sample_rate=44100, **kwargs)
        return train_set, val_set