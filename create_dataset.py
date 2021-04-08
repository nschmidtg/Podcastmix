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
import librosa

"""
Create the augmented dataset
using the VCTK and the JamendoPopular datasets, an augmented
podcast/radioshow like dataset is created
"""
# modify if necesary:
speech_path = "VCTK/wav48_silence_trimmed"
speech_metadata_path = "VCTK/speaker-info.txt"

music_path = "Jamendo/music"
music_metadata_path = "Jamendo/metadata.json"

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
    os.makedirs('podcastmix/metadata/test')


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

csv_path_tr_s = 'podcastmix/metadata/train/speech.csv'
csv_path_tr_m = 'podcastmix/metadata/train/music.csv'
csv_path_va_s = 'podcastmix/metadata/val/speech.csv'
csv_path_va_m = 'podcastmix/metadata/val/music.csv'
csv_path_te_s = 'podcastmix/metadata/test/speech.csv'
csv_path_te_m = 'podcastmix/metadata/test/music.csv'

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

# Initialize usefull arrays
counter = 0
speech_train_set = []
speech_val_set = []
speech_test_set = []
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
errors = []
# read the json and start copying the files to the respective directory.
# at the same time the csv files are being filled.
for song_id in keys:
    song = json_file.get(song_id)
    try:
        current_file_path = music_path + '/' + song['id'] + '.flac'
        print('********', current_file_path)
        audio_info = torchaudio.info(current_file_path)
        print(counter, '/', len(keys))
        print('1', current_file_path)
        exists = False
        channels = audio_info.num_channels
        if channels == 2:
            if counter < int(train_prop * len(keys)):
                # train
                print('2', current_file_path)
                destination = train_path + '/music/' + song['id'] + '.flac'
                if not os.path.exists(destination):
                    # copyfile(music_path + '/' + song['id'] + '.flac', destination)
                    # audio, original_sf = torchaudio.load(
                    #     current_file_path,
                    #     normalize = True)
                    # audio.transforms.Resample(orig_freq = original_sf, new_freq = 44100)
                    # torchaudio.save(destination, audio, sample_rate=44100, bits_per_sample=16)
                    exists = True
                song['local_path'] = destination
                print('3', current_file_path)
                music_train_set.append(song)
                csv_path = 'podcastmix/metadata/train/music.csv'
            elif counter >= int(train_prop * len(keys)) and counter < int((train_prop + val_prop) * len(keys)):
                # val
                destination = val_path + '/music/' + song['id'] + '.flac'
                if not os.path.exists(destination):
                    # copyfile(music_path + '/' + song['id'] + '.flac', destination)
                    # audio, original_sf = torchaudio.load(
                    #     current_file_path,
                    #     normalize = True)
                    # audio.transforms.Resample(orig_freq = original_sf, new_freq = 44100)
                    # torchaudio.save(destination, audio, sample_rate=44100, bits_per_sample=16)
                    exists = True
                song['local_path'] = destination
                music_val_set.append(song)
                csv_path = 'podcastmix/metadata/val/music.csv'
            else:
                # test
                destination = test_path + '/music/' + song['id'] + '.flac'
                if not os.path.exists(destination):
                    # copyfile(music_path + '/' + song['id'] + '.flac', destination)
                    # audio, original_sf = torchaudio.load(
                    #     current_file_path,
                    #     normalize = True)
                    # audio.transforms.Resample(orig_freq = original_sf, new_freq = 44100)
                    # torchaudio.save(destination, audio, sample_rate=44100, bits_per_sample=16)
                    exists = True
                song['local_path'] = destination
                music_test_set.append(song)
                csv_path = 'podcastmix/metadata/test/music.csv'
            if not exists:
                with open(csv_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    song_length = audio_info.num_frames
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
print('errores', errors)
print('keys', keys[0])

# process speech files
speech_files = []
for path, subdirs, files in os.walk(speech_path):
    for name in files:
        speech_files.append(os.path.join(path, name))

# shuffle speech
random.shuffle(speech_files)

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
counter = 0
for speech_path_dir in speech_files:
    print(counter, '/', len(speech_files))
    exists = False
    if counter < int(train_prop * len(speech_files)):
        # train
        destination = train_path + '/speech/' + speech_path_dir.split('/')[-1].split('.')[0] + '.flac'
        if not os.path.exists(destination):
            # resample from 48kHz -> 44.1kHz
            exists = True
            # audio, original_sr = torchaudio.load(speech_path_dir, normalize=True)
            # resampled_audio = torchaudio.transforms.Resample(
            #     original_sr,
            #     44100
            # )(audio)
            # torchaudio.save(
            #     filepath=destination,
            #     src=resampled_audio,
            #     sample_rate=44100,
            #     bits_per_sample=16
            # )
        # copyfile(speech_path_dir, destination)
        speech_train_set.append(destination)
        csv_path = 'podcastmix/metadata/train/speech.csv'
    elif counter >= int(train_prop * len(speech_files)) and counter < int((train_prop + val_prop) * len(speech_files)):
        # val
        destination = val_path + '/speech/' + speech_path_dir.split('/')[-1].split('.')[0] + '.flac'
        if not os.path.exists(destination):
            exists = True
            # resample from 48kHz -> 44.1kHz
            # audio, original_sr = torchaudio.load(speech_path_dir, normalize=True)
            # resampled_audio = torchaudio.transforms.Resample(
            #     original_sr,
            #     44100
            # )(audio)
            # torchaudio.save(
            #     filepath=destination,
            #     src=resampled_audio,
            #     sample_rate=44100,
            #     bits_per_sample=16
            # )
        # copyfile(speech_path_dir, destination)
        speech_val_set.append(destination)
        csv_path = 'podcastmix/metadata/val/speech.csv'
    else:
        # test
        destination = test_path + '/speech/' + speech_path_dir.split('/')[-1].split('.')[0] + '.flac'
        if not os.path.exists(destination):
            exists = True
            # resample from 48kHz -> 44.1kHz
            # audio, original_sr = torchaudio.load(speech_path_dir, normalize=True)
            # resampled_audio = torchaudio.transforms.Resample(
            #     original_sr,
            #     44100
            # )(audio)
            # torchaudio.save(
            #     filepath=destination,
            #     src=resampled_audio,
            #     sample_rate=44100,
            #     bits_per_sample=16
            # )
        # copyfile(speech_path_dir, destination)
        speech_test_set.append(destination)
        csv_path = 'podcastmix/metadata/test/speech.csv'
#    if not exists:
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        audio = torchaudio.info(destination)
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
