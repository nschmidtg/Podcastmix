from math import floor
from shutil import copyfile
from mutagen.flac import FLAC
from mutagen.wave import WAVE
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
import torchaudio
"""
Create the augmented dataset
using the VCTK and the JamendoPopular datasets, an augmented 
podcast/radioshow like dataset is created
"""
# modify if necesary:
speech_path = "../../VCTK/wav48_silence_trimmed"
speech_metadata_path = "../../VCTK/speaker-info.txt"

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
    "music_path",
    "length"
    ]

csv_path = 'podcastmix/metadata/train/speech.csv'
csv_path = 'podcastmix/metadata/train/music.csv'
csv_path = 'podcastmix/metadata/val/speech.csv'
csv_path = 'podcastmix/metadata/val/music.csv'
csv_path = 'podcastmix/metadata/test/speech.csv'
csv_path = 'podcastmix/metadata/test/music.csv'

create_csv_metadata(csv_path, speech_headers)
create_csv_metadata(csv_path, music_headers)
create_csv_metadata(csv_path, speech_headers)
create_csv_metadata(csv_path, music_headers)
create_csv_metadata(csv_path, speech_headers)
create_csv_metadata(csv_path, music_headers)

# initialize the random seed
random.seed(1)

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

# open music metadata
with open(music_metadata_path) as file:
    json_file = json.load(file)

# shuffle music
keys = list(json_file.keys())
random.shuffle(keys)
errors = []
for song_id in keys:
    song = json_file.get(song_id)
    
    # try:
    audio_info = torchaudio.info(music_path + '/' + song['id'] + '.flac')
    print(audio_info.num_channels)
    channels = audio_info.num_channels
    if channels == 2:
        if counter < int(train_prop * len(keys)):
            # train
            print(counter, '/', len(keys))
            destination = train_path + '/music/' + song['id'] + '.flac'
            # audio = librosa.load(music_path + '/' + song['id'] + '.mp3', sr=44100, mono=False)[0]
            # sf.write(destination, audio.T, samplerate=44100)
            copyfile(music_path + '/' + song['id'] + '.flac', destination)
            song['local_path'] = destination
            music_train_set.append(song)
            csv_path = 'podcastmix/metadata/train/music.csv'
        elif counter >= int(train_prop * len(keys)) and counter < int((train_prop + val_prop) * len(keys)):
            # val
            destination = val_path + '/music/' + song['id'] + '.flac'
            # audio = librosa.load(music_path + '/' + song['id'] + '.mp3', sr=44100, mono=False)[0]
            # sf.write(destination, audio.T, samplerate=44100)
            copyfile(music_path + '/' + song['id'] + '.flac', destination)
            song['local_path'] = destination
            music_val_set.append(song)
            csv_path = 'podcastmix/metadata/val/music.csv'
        else:
            # test
            destination = test_path + '/music/' + song['id'] + '.flac'
            # audio = librosa.load(music_path + '/' + song['id'] + '.mp3', sr=44100, mono=False)[0]
            # sf.write(destination, audio.T, samplerate=44100)
            copyfile(music_path + '/' + song['id'] + '.flac', destination)
            song['local_path'] = destination
            music_test_set.append(song)
            csv_path = 'podcastmix/metadata/test/music.csv'
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            song_length = floor(audio_info.sample_rate * audio_info.num_frames)
            writer.writerow([song['id'],song['id'],song['name'].replace(',',''),song['artist_name'].replace(',',''),song['album_name'].replace(',',''),song['license_ccurl'],song['releasedate'],song['local_path'] ,song_length])
    else:
        errors.append(song_id)
    # except:
    #     errors.append(song_id)
    counter +=1
print('errores',errors)
print('keys',keys[0])

# process speech files
speech_files = np.array([])
for path, subdirs, files in os.walk(speech_path):
   for name in files:
       speech_files = np.append(speech_files, os.path.join(path, name))

# shuffle speech
np.random.shuffle(speech_files)

# read speech metadata.txt
speaker_params = {}
s_m = open(speech_metadata_path, 'r')
lines = s_m.readlines()
count = 0
for line in lines:
    if count != 0:
       #skip headers
       cols = re.split('\s+', line)
       speaker_params[cols[0]] = {'speaker_id':cols[0],'speaker_age':cols[1],'speaker_gender':cols[2],'speaker_accent':cols[3]}
    count += 1

counter = 0
for speech_path_dir in speech_files:
    if counter < int(train_prop * len(speech_files)):
        # train
        destination = train_path + '/speech/' + speech_path_dir.split('/')[-1]
        # resample from 48kHz -> 44.1kHz
        audio = librosa.load(speech_path_dir, sr=44100)[0]
        sf.write(destination, audio, samplerate=44100)
        # copyfile(speech_path_dir, destination)
        speech_train_set.append(destination)
        csv_path = 'podcastmix/metadata/train/speech.csv'
    elif counter >= int(train_prop * len(speech_files)) and counter < int((train_prop + val_prop) * len(speech_files)):
        # val
        destination = val_path + '/speech/' + speech_path_dir.split('/')[-1]
        audio = librosa.load(speech_path_dir, sr=44100)[0]
        sf.write(destination, audio, samplerate=44100)
        speech_val_set.append(destination)
        csv_path = 'podcastmix/metadata/val/speech.csv'
    else:
        # test
        destination = test_path + '/speech/' + speech_path_dir.split('/')[-1]
        audio = librosa.load(speech_path_dir, sr=44100)[0]
        sf.write(destination, audio, samplerate=44100)
        speech_test_set.append(destination)
        csv_path = 'podcastmix/metadata/test/speech.csv'
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        audio = FLAC(destination)
        element_length = floor(audio.info.sample_rate * audio.info.length)
        speech_cmp = destination.split('/')[-1].split('_')
        params = speaker_params[speech_cmp[0]]
        writer.writerow([speech_cmp[1]+'_'+speech_cmp[2].split('.')[0], speech_cmp[0], params['speaker_age'], params['speaker_gender'], params['speaker_accent'].replace(',',''), destination, element_length])
    counter += 1