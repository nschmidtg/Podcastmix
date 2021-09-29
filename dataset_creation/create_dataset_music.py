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
sys.path.append('../utils')
from resample_and_copy import resample_and_copy

"""
Create the augmented dataset
using the VCTK and the JamendoPopular datasets, an augmented
podcast/radioshow like dataset is created
"""
# modify if necesary:
music_path = "../Jamendo/music"
music_metadata_path = "../Jamendo/metadata.json"

# create files structure
root_dir = '../podcastmix-correct/podcastmix-synth'
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

def create_folder_structure(path):
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + '/music', exist_ok=True)

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

csv_path_tr_m = os.path.join(metadata_path, 'train/music.csv')
csv_path_va_m = os.path.join(metadata_path, 'val/music.csv')
csv_path_te_m = os.path.join(metadata_path, 'test/music.csv')

create_csv_metadata(csv_path_tr_m, music_headers)
create_csv_metadata(csv_path_va_m, music_headers)
create_csv_metadata(csv_path_te_m, music_headers)

# initialize the random seed
random.seed(1)

# determine the train/test partition
train_prop = 0.8
val_prop = 0.1
test_prop = 0.1

# Initialize usefull arrays
counter = 0
music_train_set = []
music_val_set = []
music_test_set = []

# process music files
# open music metadata
with open(music_metadata_path) as file:
    json_file = json.load(file)

# shuffle music
keys = list(json_file.keys())
random.shuffle(keys)

# create a dict for the artists
artists = {}
for song_id in keys:
    song = json_file.get(song_id)
    artist_id = song['artist_id']
    if artist_id in artists.keys():
        artists[artist_id].append(song_id)
    else:
        artists[artist_id] = [song_id]


errors = []
destination_sr = 44100
# read the json and start copying the files to the respective directory.
# at the same time the csv files are being filled.
artists_counter = 0
# print(artists)

for artist_id in artists.keys():
    song_id_array = artists[artist_id]
    for song_id in song_id_array:
        song = json_file.get(song_id)
        try:
            current_file_path = music_path + '/' + song['id'] + '.flac'
            audio_info = torchaudio.info(current_file_path)
            print(counter, '/', len(keys))
            exists = False
            channels = audio_info.num_channels
            if channels == 2:
                if artists_counter < int(train_prop * len(artists)):
                    # train
                    destination = train_path + '/music/' + song['id'] + '.flac'
                    song['local_path'] = destination
                    csv_path = csv_path_tr_m
                elif artists_counter >= int(train_prop * len(artists)) and artists_counter < int((train_prop + val_prop) * len(artists)):
                    # val
                    destination = val_path + '/music/' + song['id'] + '.flac'
                    song['local_path'] = destination
                    csv_path = csv_path_va_m
                else:
                    # test
                    destination = test_path + '/music/' + song['id'] + '.flac'
                    song['local_path'] = destination
                    csv_path = csv_path_te_m
                audio = resample_and_copy(current_file_path, destination, destination_sr)
                with open(csv_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    song_length = audio.shape[-1]
                    # flatten tags
                    tags = json.dumps(song["musicinfo"]["tags"])
                    writer.writerow(
                        [
                            song['id'],
                            song['id'],
                            song['name'].replace(',', ''),
                            song['artist_name'].replace(',', ''),
                            song['album_name'].replace(',', ''),
                            song['license_ccurl'],
                            song['releasedate'],
                            song["image"],
                            song["musicinfo"]["vocalinstrumental"],
                            song["musicinfo"]["lang"],
                            song["musicinfo"]["gender"],
                            song["musicinfo"]["acousticelectric"],
                            song["musicinfo"]["speed"],
                            tags.replace(',', '/'),
                            song['local_path'],
                            song_length])
            else:
                errors.append(song_id)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            errors.append(song_id)
        counter += 1
    artists_counter += 1
print('errores', errors)
print('keys', keys[0])
