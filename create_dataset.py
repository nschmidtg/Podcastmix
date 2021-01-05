# MiniLibriMix is a tiny version of LibriMix (https://github.com/JorisCos/LibriMix),
# which is a free speech separation dataset.
from asteroid.data import LibriMix

# import musdb to create the mixtures: https://github.com/sigsep/sigsep-mus-db
import musdb

# Asteroid's System is a convenience wrapper for PyTorch-Lightning.
from asteroid.engine import System

from IPython.display import display, Audio
import soundfile as sf
import librosa, os
# download the musdb library
mus = musdb.DB(download=True)

# This will automatically download MiniLibriMix from Zenodo on the first run.
train_loader, val_loader = LibriMix.loaders_from_mini(task="sep_clean", batch_size=8)

"""
Create the augmented dataset
using the LibriMix and the Musdb18 datasets, an augmented 
podcast/radioshow like dataset is created
"""

def create_folder_structure(path):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + '/linear_mono'):
        os.makedirs(path + '/linear_mono')
    if not os.path.exists(path + '/linear_stereo'):
        os.makedirs(path + '/linear_stereo')
    if not os.path.exists(path + '/sidechain_mono'):
        os.makedirs(path + '/sidechain_mono')
    if not os.path.exists(path + '/sidechain_stereo'):
        os.makedirs(path + '/sidechain_stereo')
    if not os.path.exists(path + '/track_mono'):
        os.makedirs(path + '/track_mono')
    if not os.path.exists(path + '/track_stereo'):
        os.makedirs(path + '/track_stereo')
    if not os.path.exists(path + '/speech_mono'):
        os.makedirs(path + '/speech_mono')

# create files structure
train_path = "augmented_dataset/train"
create_folder_structure(train_path)

val_path = "augmented_dataset/val"
create_folder_structure(val_path)

# create the metadata directory
if not os.path.exists('augmented_dataset/metadata'):
    os.makedirs('augmented_dataset/metadata')
if not os.path.exists('augmented_dataset/metadata/train'):
    os.makedirs('augmented_dataset/metadata/train')
if not os.path.exists('augmented_dataset/metadata/val'):
    os.makedirs('augmented_dataset/metadata/val')

from os import listdir
from os.path import isfile, join
import random
import numpy as np
import re
import csv

def mix_audio_sources(track_path, speech_path, output_path, music_to_speech_ratio = 0.2):
    """
    Creates 4 mixes for the a music and a speech track and locates it in the output_path
    the 4 mixes are: linear_mono, linear_stereo, sidechain_mono, sidechain_stereo
    librimix is mono and musdb stereo
    """
    # read the files
    track = librosa.load(track_path, sr=44100, mono=False)[0]
    speech = librosa.load(speech_path, sr=44100)[0]
    # match the length of the files
    min_lenght = min(len(track[0]), len(speech))
    
    # crop the files to match in length
    cropped_track_stereo = np.array([track[0][0:min_lenght], track[1][0:min_lenght]])
    cropped_track_mono = cropped_track_stereo[0] + cropped_track_stereo[1]
    cropped_speech = speech[0:min_lenght]
    
    linear_stereo = cropped_track_stereo * music_to_speech_ratio + cropped_speech
    linear_mono = cropped_track_mono * music_to_speech_ratio + cropped_speech
    
    # write the files
    file_name = re.sub("[^0-9a-zA-Z]+", "-", track_path.split('/')[-1]) + '_' + speech_path.split('/')[-1]
    sf.write(output_path + "/linear_mono/" + file_name, linear_mono, 44100, subtype='PCM_24')
    sf.write(output_path + "/linear_stereo/" + file_name, linear_stereo.T, 44100, subtype='PCM_24')
    sf.write(output_path + "/speech_mono/" + file_name, cropped_speech, 44100, subtype='PCM_24')
    sf.write(output_path + "/track_mono/" + file_name, cropped_track_mono, 44100, subtype='PCM_24')
    sf.write(output_path + "/track_stereo/" + file_name, cropped_track_stereo.T, 44100, subtype='PCM_24')

    return file_name, min_lenght

# I used the s1 source from the MiniLibriMix for the train set
# and the s3 source for the val set
speech_path_train = "MiniLibriMix/val/s1/"
speech_path_val = "MiniLibriMix/val/s2/"
    
speech_array_train = [f for f in listdir(speech_path_train) if isfile(join(speech_path_train, f))]
speech_array_val = [f for f in listdir(speech_path_val) if isfile(join(speech_path_val, f))]

# initialize the random seed
random.seed(1)

# create the train csv file
csv_path = 'augmented_dataset/metadata/train/linear_stereo.csv'
with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["","mixture_ID","mixture_path","track_path","speech_path","length"])

csv_path = 'augmented_dataset/metadata/train/linear_mono.csv'
with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["","mixture_ID","mixture_path","track_path","speech_path","length"])

# create the val csv file
csv_path = 'augmented_dataset/metadata/val/linear_stereo.csv'
with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["","mixture_ID","mixture_path","track_path","speech_path","length"])

csv_path = 'augmented_dataset/metadata/val/linear_mono.csv'
with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["","mixture_ID","mixture_path","track_path","speech_path","length"])

# create the train/val sets:

# 100 train, 44 val
n_train = 100
i = 0
for track in mus:
    
    track_file = track.path
    if i < n_train:
        # creating the training set for the first n_train songs
        speech_name = speech_array_train[random.randint(0,len(speech_array_train)-1)]
        path = train_path
        csv_path = 'augmented_dataset/metadata/train'
        speech_path = speech_path_train
        
    else:
        # the val set
        speech_name = speech_array_val[random.randint(0,len(speech_array_val)-1)]
        path = val_path
        csv_path = 'augmented_dataset/metadata/val'
        speech_path = speech_path_val

    # path of the speech
    speech_file = speech_path + speech_name
    file_name, min_length = mix_audio_sources(track_file, speech_file, path, music_to_speech_ratio= 0.1)

    # add a new row to the corresponding csv metadata file
    csv_path_file = csv_path + '/linear_mono.csv'
    with open(csv_path_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
                        i,
                        file_name,
                        path + "/linear_mono/" + file_name,
                        path + "/track_mono/" + file_name,
                        path + "/speech_mono/" + file_name,
                        min_length
            ])
    
    csv_path_file = csv_path + '/linear_stereo.csv'
    with open(csv_path_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
                        i,
                        file_name,
                        path + "/linear_stereo/" + file_name,
                        path + "/track_stereo/" + file_name,
                        path + "/speech_stereo/" + file_name,
                        min_length
            ])
    
    i += 1