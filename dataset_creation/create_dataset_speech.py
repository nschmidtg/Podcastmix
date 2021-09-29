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
sys.path.append('utils')
from resample_and_copy import resample_and_copy

"""
Create the augmented dataset
using the VCTK and the JamendoPopular datasets, an augmented
podcast/radioshow like dataset is created
"""
# modify if necesary:
speech_path = "VCTK/wav48_silence_trimmed"
speech_metadata_path = "VCTK/speaker-info.txt"

# create files structure
root_dir = 'podcastmix/podcastmix-synth'
if not os.path.exists(root_dir):
    os.makedirs(root_dir)


def create_folder_structure(path):
    if not os.path.exists(path):
        os.makedirs(path)
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
os.makedirs(metadata_path, exist_ok=True)
os.makedirs(os.path.join(metadata_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(metadata_path, 'test'), exist_ok=True)
os.makedirs(os.path.join(metadata_path, 'val'), exist_ok=True)


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

csv_path_tr_s = os.path.join(metadata_path, 'train/speech.csv')
csv_path_va_s = os.path.join(metadata_path, 'val/speech.csv')
csv_path_te_s = os.path.join(metadata_path, 'test/speech.csv')

create_csv_metadata(csv_path_tr_s, speech_headers)
create_csv_metadata(csv_path_va_s, speech_headers)
create_csv_metadata(csv_path_te_s, speech_headers)

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

# list all subdirectories in VCTK:
speakers = [f for f in listdir(speech_path)]
destination_sr = 44100
counter = 0
for i, speaker in enumerate(speakers):
    print(i, '/', len(speakers), 'speakers')
    for path, subdirs, files in os.walk(os.path.join(speech_path, speaker)):
        for name in files:
            speech_path_dir = os.path.join(path, name)
            # arbitrarily pick mic1
            if 'mic1' in speech_path_dir:
                if i < int(train_prop * len(speakers)):
                    # train
                    destination = train_path + '/speech/' + speech_path_dir.split('/')[-1].split('.')[0] + '.flac'
                    csv_path = csv_path_tr_s
                elif i >= int(train_prop * len(speakers)) and i < int((train_prop + val_prop) * len(speakers)):
                    # val
                    destination = val_path + '/speech/' + speech_path_dir.split('/')[-1].split('.')[0] + '.flac'
                    csv_path = csv_path_va_s
                else:
                    # test
                    destination = test_path + '/speech/' + speech_path_dir.split('/')[-1].split('.')[0] + '.flac'
                    csv_path = csv_path_te_s
                audio = resample_and_copy(speech_path_dir, destination, destination_sr)
                with open(csv_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    element_length = audio.shape[-1]
                    # print(element_length)
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
