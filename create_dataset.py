import soundfile as sf
import librosa, os
from os import listdir
from os.path import isfile, join
import random
import numpy as np
import re
import csv
# download the JamendoPopular library   
# TODO wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt 
# download the VCTK
# TODO !wget --no-check-certificate 'https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip?sequence=2&isAllowed=y' -O 'VCTK-Corpus-0.92.zip'
"""
Create the augmented dataset
using the VCTK and the JamendoPopular datasets, an augmented 
podcast/radioshow like dataset is created
"""
# create files structure
if not os.path.exists('podcastmix'):
        os.makedirs('podcastmix')

def create_folder_structure(path):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + '/speech'):
        os.makedirs(path + '/music')

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

from os import listdir
from os.path import isfile, join
import random
import numpy as np
import re
import csv

# I used the s1 source from the MiniLibriMix for the train set
# and the s3 source for the val and test set
speech_path_train = "MiniLibriMix/val/s1/"
speech_path_val_test = "MiniLibriMix/val/s2/"
    
speech_array_train = [f for f in listdir(speech_path_train) if isfile(join(speech_path_train, f))]
speech_array_val_test = [f for f in listdir(speech_path_val_test) if isfile(join(speech_path_val_test, f))]

# initialize the random seed
random.seed(1)
# shuffle
random.shuffle(speech_array_train)
random.shuffle(speech_array_val_test)



def create_csv_metadata(csv_path, headers):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
# create the train csv file

speech_headers = ["speech_ID","speeker_id","speech_path","length"]
music_headers = ["music_ID", "music_path","length"]

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

# 70% train, 15 test 15 val
# TODO I have to split the two datasets in the respective folder creating the 2 metadata files
n_train = 102
n_val_test = 21

# speech_array_train = speech_array_train[0:n_train]
# speech_array_val_test = speech_array_train[0: 2 * n_val_test]
# print("speech_array_val_test", len(speech_array_val_test))

# i = 0
# for track in mus:
    
#     track_file = track.path
#     if i < n_train:
#         # creating the training set for the first n_train songs
#         csv_path = 'podcastmix/metadata/train'
#         speech_path = speech_path_train
#         path = train_path
#         speech_name = speech_array_train[i]
        
#     else:
#         # the val/test set
#         if(i - n_train < n_val_test):
#             csv_path = 'podcastmix/metadata/val'
#             speech_path = speech_path_val_test
#             path = val_path
#         else:
#             csv_path = 'podcastmix/metadata/test'
#             speech_path = speech_path_val_test
#             path = test_path
#         print("i",i)
#         print("n_train", n_train)
#         print("//////////////////////////")
#         speech_name = speech_array_val_test[i - n_train - 1]

#     # path of the speech
#     speech_file = speech_path + speech_name
#     file_name, min_length = mix_audio_sources(track_file, speech_file, path, music_to_speech_ratio = 0.1)

#     # add a new row to the corresponding csv metadata file
#     csv_path_file = csv_path + '/speech.csv'
#     with open(csv_path_file, 'a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([
#                         i,
#                         file_name,
#                         path + "/linear_mono/" + file_name,
#                         path + "/music_mono/" + file_name,
#                         path + "/speech_mono/" + file_name,
#                         min_length
#             ])

#     # add a new row to the corresponding csv metadata file
#     csv_path_file = csv_path + '/music.csv'
#     with open(csv_path_file, 'a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([
#                         i,
#                         file_name,
#                         path + "/linear_mono/" + file_name,
#                         path + "/music_mono/" + file_name,
#                         path + "/speech_mono/" + file_name,
#                         min_length
#             ])
    
#     i += 1
