import soundfile as sf
import librosa, os
from os import listdir
from os.path import isfile, join
import random
import numpy as np
import re
import sys
import json
import csv
# download the JamendoPopular library   
# TODO wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-0aH0V9VD1leaHVmrbI6jNbQyz7Lwp_6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-0aH0V9VD1leaHVmrbI6jNbQyz7Lwp_6" -O JamendoPopular && rm -rf /tmp/cookies.txt 
# download the VCTK
# TODO !wget --no-check-certificate 'https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip?sequence=2&isAllowed=y' -O 'VCTK-Corpus-0.92.zip'
"""
Create the augmented dataset
using the VCTK and the JamendoPopular datasets, an augmented 
podcast/radioshow like dataset is created
"""
# modify if necesary:
speech_path = "../../VCTK/wav48_silence_trimmed"
speech_metadata_path = "../../VCTK/speaker-info.txt"

music_path = "../../JamendoBoost/music"
music_metadata_path = "../../JamendoBoost/metadata.json"

# create files structure
if not os.path.exists('podcastmix'):
        os.makedirs('podcastmix')

def create_folder_structure(path):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + '/music'):
        os.makedirs(path + '/music')
    if not os.path.exists(path + '/speech'):
        os.makedirs(path + '/speech')

# create files structure
train_path = "podcastmix/train"
create_folder_structure(train_path)

val_path = "podcastmix/val"
create_folder_structure(val_path)

test_path = "podcastmix/test"
create_folder_structure(test_path)

# create the metadata directory
if not os.path.exists('podcastmix/metadata'):
    os.makedirs('podcastmix/metadata')
if not os.path.exists('podcastmix/metadata/train'):
    os.makedirs('podcastmix/metadata/train')
if not os.path.exists('podcastmix/metadata/val'):
    os.makedirs('podcastmix/metadata/val')
if not os.path.exists('podcastmix/metadata/test'):
    os.makedirs('podcastmix/metadata/tesst')

def create_csv_metadata(csv_path, headers):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
# create the train csv file

speech_headers = [
    "speech_ID",
    "speaker_age",
    "speaker_gender",
    "speaker_accent",
    "speaker_region_comment",
    "speech_file_path",
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
    "music_path",
    "length"
    ]

csv_path = 'podcastmix/metadata/train/speech.csv'
create_csv_metadata(csv_path, speech_headers)

csv_path = 'podcastmix/metadata/train/music.csv'
create_csv_metadata(csv_path, music_headers)

# create the val csv file
csv_path = 'podcastmix/metadata/val/speech.csv'
create_csv_metadata(csv_path, speech_headers)

csv_path = 'podcastmix/metadata/val/music.csv'
create_csv_metadata(csv_path, music_headers)

# create the test csv file
csv_path = 'podcastmix/metadata/test/speech.csv'
create_csv_metadata(csv_path, speech_headers)

csv_path = 'podcastmix/metadata/test/music.csv'
create_csv_metadata(csv_path, music_headers)

# initialize the random seed
random.seed(1)

train_prop = 0.8
val_prop = 0.1
test_prop = 0.1

counter = 0
speech_train_set = []
speech_val_set = []
speech_test_set = []
music_train_set = []
music_val_set = []
music_test_set = []

with open(music_metadata_path) as file:
    json_file = json.load(file)

for song_id in json_file:
    song = json_file.get(song_id)
    if counter < int(train_prop * len(json_file)):
        # train
        music_train_set.append(song)
    elif counter >= int(train_prop * len(json_file)) and counter < int((train_prop + val_prop) * len(json_file)):
        # val
        music_val_set.append(song)
    else:
        # test
        music_test_set.append(song)
    counter += 1

print(len(music_train_set))
print(len(music_val_set))
print(len(music_test_set))

speech_files = np.array([])
for path, subdirs, files in os.walk(speech_path):
    for name in files:
        speech_files = np.append(speech_files, os.path.join(path, name))

counter = 0
for speech_path in speech_files:
    speaker_id = speech_path.split('_')[0]
    if counter < int(train_prop * len(speech_files)):
        # train
        speech_train_set.append(speech_path)
    elif counter >= int(train_prop * len(speech_files)) and counter < int((train_prop + val_prop) * len(speech_files)):
        # val
        speech_val_set.append(speech_path)
    else:
        # test
        speech_test_set.append(speech_path)
    counter += 1

print(len(speech_train_set))
print(len(speech_val_set))
print(len(speech_test_set))

import re
s_m = open(speech_metadata_path, 'r')
lines = s_m.readlines()
for line in lines:
    print(re.split('\s+', line))


sys.exit()


sets = [
    [speech_train_set, 'podcastmix/metadata/train/speech.csv'],
    [speech_val_set, 'podcastmix/metadata/val/speech.csv'],
    [speech_test_set, 'podcastmix/metadata/test/speech.csv'],
    [music_train_set, 'podcastmix/metadata/train/music.csv'],
    [music_val_set, 'podcastmix/metadata/val/music.csv'],
    [music_test_set, 'podcastmix/metadata/test/music.csv']
]
i=0
for set, csv_path in sets:
    for element in set:
        # add a new row to the corresponding csv metadata file
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            # speech_ID","speech_path","length"]
            element_length = len(sf.read(element)[0])
            writer.writerow([
                            i,
                            element,
                            element_length
                ])
            i += 1
