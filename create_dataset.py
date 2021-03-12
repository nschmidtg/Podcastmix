from math import floor
from shutil import copyfile
from mutagen.wave import WAVE
from mutagen.mp3 import MP3
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

#csv_path = 'podcastmix/metadata/train/speech.csv'
#create_csv_metadata(csv_path, speech_headers)

csv_path = 'podcastmix/metadata/train/music.csv'
create_csv_metadata(csv_path, music_headers)

# create the val csv file
#csv_path = 'podcastmix/metadata/val/speech.csv'
#create_csv_metadata(csv_path, speech_headers)

csv_path = 'podcastmix/metadata/val/music.csv'
create_csv_metadata(csv_path, music_headers)

# create the test csv file
#csv_path = 'podcastmix/metadata/test/speech.csv'
#create_csv_metadata(csv_path, speech_headers)

csv_path = 'podcastmix/metadata/test/music.csv'
create_csv_metadata(csv_path, music_headers)

# initialize the random seed
random.seed(1)

train_prop = 0.8
val_prop = 0.1
test_prop = 0.1

counter = 0
#speech_train_set = []
#speech_val_set = []
#speech_test_set = []
music_train_set = []
music_val_set = []
music_test_set = []

with open(music_metadata_path) as file:
    json_file = json.load(file)

# shuffle music
keys = list(json_file.keys())
random.shuffle(keys)
errors = []

pre_errors = ['1803718', '1822339', '1802304', '1791375', '1800490', '1822260', '1823100', '1822678', '1821959', '1822567', '1829588', '1822734', ' 1822948', '1822951', '1800335', '1810679', '1822534', '1829175', '1804728', '1800491', '1800353', '1829167', '1809521']

for song_id in keys:
    song = json_file.get(song_id)
    print(counter, '/', len(keys))
    if not song_id in pre_errors:
        try:
            if counter < int(train_prop * len(keys)):
                # train
                destination = train_path + '/music/' + song['id'] + '.wav'
                # audio = librosa.load(music_path + '/' + song['id'] + '.mp3', sr=44100, mono=False)[0]
                # sf.write(destination, audio.T, samplerate=44100)
                # copyfile(music_path + '/' + song['id'] + '.mp3', destination)
                song['local_path'] = destination
                music_train_set.append(song)
            elif counter >= int(train_prop * len(keys)) and counter < int((train_prop + val_prop) * len(keys)):
                # val
                destination = val_path + '/music/' + song['id'] + '.wav'
                # audio = librosa.load(music_path + '/' + song['id'] + '.mp3', sr=44100, mono=False)[0]
                # sf.write(destination, audio.T, samplerate=44100)
                # copyfile(music_path + '/' + song['id'] + '.mp3', destination)
                song['local_path'] = destination
                music_val_set.append(song)
            else:
                # test
                destination = test_path + '/music/' + song['id'] + '.wav'
                # audio = librosa.load(music_path + '/' + song['id'] + '.mp3', sr=44100, mono=False)[0]
                # sf.write(destination, audio.T, samplerate=44100)
                # copyfile(music_path + '/' + song['id'] + '.mp3', destination)
                song['local_path'] = destination
                music_test_set.append(song)
            counter +=1
        except:
            errors.append(song_id)

print('errores',errors)
print('keys',keys[0])
#speech_files = np.array([])
#for path, subdirs, files in os.walk(speech_path):
#    for name in files:
#        speech_files = np.append(speech_files, os.path.join(path, name))

# shuffle speech
#np.random.shuffle(speech_files)

#counter = 0
#for speech_path in speech_files:
#    if counter < int(train_prop * len(speech_files)):
#        # train
#        destination = train_path + '/speech/' + speech_path.split('/')[-1].split('.')[0] + '.wav'
#        audio = librosa.load(speech_path, sr=44100)[0]
#        sf.write(destination, audio, samplerate=44100)
#        # copyfile(speech_path, destination)
#        speech_train_set.append(destination)
#    elif counter >= int(train_prop * len(speech_files)) and counter < int((train_prop + val_prop) * len(speech_files)):
#        # val
#        destination = val_path + '/speech/' + speech_path.split('/')[-1].split('.')[0] + '.wav'
#        audio = librosa.load(speech_path, sr=44100)[0]
#        sf.write(destination, audio, samplerate=44100)
#        speech_val_set.append(destination)
#    else:
#        # test
#        destination = test_path + '/speech/' + speech_path.split('/')[-1].split('.')[0] + '.wav'
#        audio = librosa.load(speech_path, sr=44100)[0]
#        sf.write(destination, audio, samplerate=44100)
#        speech_test_set.append(destination)
#    counter += 1

# read speech metadata.txt
#import re
#speaker_params = {}
#s_m = open(speech_metadata_path, 'r')
#lines = s_m.readlines()
#count = 0
#for line in lines:
#    if count != 0:
#        #skip headers
#        cols = re.split('\s+', line)
#        speaker_params[cols[0]] = {'speaker_id':cols[0],'speaker_age':cols[1],'speaker_gender':cols[2],'speaker_accent':cols[3]}
#    count += 1

sets = [
#    [speech_train_set, 'podcastmix/metadata/train/speech.csv'],
#    [speech_val_set, 'podcastmix/metadata/val/speech.csv'],
#    [speech_test_set, 'podcastmix/metadata/test/speech.csv'],
    [music_train_set, 'podcastmix/metadata/train/music.csv'],
    [music_val_set, 'podcastmix/metadata/val/music.csv'],
    [music_test_set, 'podcastmix/metadata/test/music.csv']
]

i=0

errors2=[]
for set, csv_path in sets:
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        for element in set:
            print(i)
            try:
                if 'music' in csv_path:
                    audio = WAVE(element['local_path'])
                    element_length = floor(audio.info.sample_rate * audio.info.length)
                    writer.writerow([element['id'],element['id'],element['name'],element['artist_name'],element['album_name'],element['license_ccurl'],element['releasedate'],element['local_path'] ,element_length])
                elif 'speech' in csv_path:
                    audio = WAVE(element)
                    element_length = floor(audio.info.sample_rate * audio.info.length)
                    speech_cmp = element.split('/')[-1].split('_')
                    params = speaker_params[speech_cmp[0]]
                    writer.writerow([speech_cmp[1]+'_'+speech_cmp[2].split('.')[0], speech_cmp[0], params['speaker_age'], params['speaker_gender'], params['speaker_accent'], element, element_length])
                i += 1
            except:
                errors2.append(element['id'])
print(errors2)
