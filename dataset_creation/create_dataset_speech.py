import sys
import soundfile as sf
from shutil import copyfile
import os
from os import listdir
from os.path import isfile, join
import random
import numpy as np
import re
import sys
import json
import csv
import torchaudio

"""
Create the augmented dataset
using the VCTK and the JamendoPopular datasets, an augmented
podcast/radioshow like dataset is created
"""
# modify if necesary:
speech_path = "../VCTK/wav48_silence_trimmed"
speech_metadata_path = "../VCTK/speaker-info.txt"

music_path = "../Jamendo/music"
music_metadata_path = "../Jamendo/metadata.json"

# create files structure
root_dir = '../podcastmix-correct/podcastmix-synth'
if not os.path.exists(root_dir):
    os.makedirs(root_dir)


def create_folder_structure(path):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + '/music'):
        os.makedirs(path + '/music')
    if not os.path.exists(path + '/speech'):
        os.makedirs(path + '/speech')


# create files structure
train_path = os.path.join(root_dir, 'train')
create_folder_structure(train_path)

val_path = os.path.join(root_dir, 'val')
create_folder_structure(val_path)

test_path = os.path.join(root_dir, 'test')
create_folder_structure(test_path)

# create the metadata directory
metadata_path = os.path.join(root_dir, 'metadata')
os.makedirs(metadata_path, exist_ok=False)
os.makedirs(os.path.join(metadata_path, 'train'), exist_ok=False)
os.makedirs(os.path.join(metadata_path, 'test'), exist_ok=False)
os.makedirs(os.path.join(metadata_path, 'val'), exist_ok=False)


def create_csv_metadata(csv_path, headers):
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)


# create the train csv file
speech_headers = [
    "speech_ID",
    "speaker_id",
    "speaker_age",
    "speaker_gender",
    "speaker_accent",
    "speech_path",
    "length"
    ]
music_headers = [
    "music_ID",
    "jamendo_id",
    "name",
    "artist_name",
    "album_name",
    "license_ccurl",
    "releasedate",
    "image",
    "vocalinstrumental",
    "lang",
    "gender",
    "acousticelectric",
    "speed",
    "tags",
    "music_path",
    "length"
]

csv_path_tr_s = os.path.join(metadata_path, 'train/speech.csv')
csv_path_tr_m = os.path.join(metadata_path, 'train/music.csv')
csv_path_va_s = os.path.join(metadata_path, 'val/speech.csv')
csv_path_va_m = os.path.join(metadata_path, 'val/music.csv')
csv_path_te_s = os.path.join(metadata_path, 'test/speech.csv')
csv_path_te_m = os.path.join(metadata_path, 'test/music.csv')

create_csv_metadata(csv_path_tr_s, speech_headers)
create_csv_metadata(csv_path_tr_m, music_headers)
create_csv_metadata(csv_path_va_s, speech_headers)
create_csv_metadata(csv_path_va_m, music_headers)
create_csv_metadata(csv_path_te_s, speech_headers)
create_csv_metadata(csv_path_te_m, music_headers)

# initialize the random seed
random.seed(1)

# determine the train/test partition
train_prop = 0.8
val_prop = 0.1
test_prop = 0.1

# read speech metadata.txt
# get the speakers metadata from the csv:
speaker_params = {}
s_m = open(speech_metadata_path, 'r')
lines = s_m.readlines()
count = 0
for line in lines:
    if count != 0:
        # skip headers
        cols = re.split('\s+', line)
        speaker_params[cols[0]] = {
            'speaker_id': cols[0],
            'speaker_age': cols[1],
            'speaker_gender': cols[2],
            'speaker_accent': cols[3]
        }
    count += 1

# iterate over the speakers downsampling and normalizing the audio.
# The new 44.1hKz versions are then written in the respective directory
# inside the podcastmix. the metadata csv for the podcastmix is also filled


def resample_and_copy(destination, destination_sr):
    exists = True
    if not os.path.exists(destination):
        # resample from 48kHz -> 44.1kHz
        exists = False
        audio, original_sr = torchaudio.load(speech_path_dir, normalize=True)
        if not original_sr == destination_sr:
            audio = torchaudio.transforms.Resample(
                original_sr,
                destination_sr
            )(audio)
        torchaudio.save(
            filepath=destination,
            src=audio,
            sample_rate=destination_sr,
            bits_per_sample=16
        )
    # copyfile(speech_path_dir, destination)
    
    return audio, exists

# list all subdirectories in VCTK:
speakers = [f for f in listdir(speech_path)][10]
destination_sr = 44100
counter = 0
for i, speaker in enumerate(speakers):
    print(i, '/', len(speakers), 'speakers')
    speech_files = []
    for path, subdirs, files in os.walk(os.path.join(speech_path, speaker)):
        for name in files:
            speech_path_dir = os.path.join(path, name)
            if counter < int(train_prop * len(speakers)):
                # train
                destination = train_path + '/speech/' + speech_path_dir.split('/')[-1].split('.')[0] + '.flac'
                csv_path = '../podcastmix/metadata/train/speech.csv'
            elif counter >= int(train_prop * len(speakers)) and counter < int((train_prop + val_prop) * len(speakers)):
                # val
                destination = val_path + '/speech/' + speech_path_dir.split('/')[-1].split('.')[0] + '.flac'
                csv_path = '../podcastmix/metadata/val/speech.csv'
            else:
                # test
                destination = test_path + '/speech/' + speech_path_dir.split('/')[-1].split('.')[0] + '.flac'
                csv_path = '../podcastmix/metadata/test/speech.csv'
            audio, exists = resample_and_copy(destination, destination_sr)
            # copyfile(speech_path_dir, destination)
            if not exists:
                with open(csv_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    element_length = audio.num_frames
                    speech_cmp = destination.split('/')[-1].split('_')
                    params = speaker_params[speech_cmp[0]]
                    writer.writerow(
                        [
                            speech_cmp[0] + '_' + speech_cmp[1] + '_' + speech_cmp[2].split('.')[0],
                            speech_cmp[0],
                            params['speaker_age'],
                            params['speaker_gender'],
                            params['speaker_accent'].replace(',', ''),
                            destination,
                            element_length
                        ]
                    )
                counter += 1
