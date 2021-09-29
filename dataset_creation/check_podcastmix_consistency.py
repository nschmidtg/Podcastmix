import os
from os import listdir
from os.path import isfile, join
from csv import reader
import torchaudio


def check_files_against_csv(csv_path, files_path, index_of_path_in_csv=7):
    not_missing = []
    format_error = {}
    format_error['sr'] = []
    format_error['bits_per_sample'] = []
    format_error['num_channels'] = []
    with open(csv_path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header is not None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                path = row[index_of_path_in_csv]
                # print(path)
                # add to array to check metadata against list of files
                if os.path.isfile(path):
                    not_missing.append(path.split('/')[5])
                else:
                    print("im not file", path)
                # check channels sr and bit depth
                info = torchaudio.info(path)
                # print(info.sample_rate, info.bits_per_sample, info.num_channels)
                if(not info.sample_rate == 44100):
                    format_error['sr'].append(path)
                if(not info.bits_per_sample == 16):
                    format_error['bits_per_sample'].append(path)
                if(('music' in path) and (not info.num_channels == 2)):
                    format_error['num_channels_music'].append(path)
                if(('speech' in path) and (not info.num_channels == 1)):
                    format_error['num_channels_speech'].append(path)

    onlyfiles = [f for f in listdir(files_path) if isfile(join(files_path, f))]
    print('Check diff between:', files_path, csv_path)
    diff = list(set(onlyfiles) - set(not_missing))
    print('List of files minus list in csv:', len(diff))
    diff2 = list(set(not_missing) - set(onlyfiles))
    print('List in csv minus list of files:', len(diff2))

    print('format errors:', format_error)

# check consistency of the dataset:
root_dir = '../podcastmix-correct/podcastmix-synth'
files_path = os.path.join(root_dir, 'test/music')
csv_path = os.path.join(root_dir, 'metadata/test/music.csv')
check_files_against_csv(csv_path, files_path, 14)

files_path = os.path.join(root_dir, 'val/music')
csv_path = os.path.join(root_dir, 'metadata/val/music.csv')
check_files_against_csv(csv_path, files_path, 14)

files_path = os.path.join(root_dir, 'train/music')
csv_path = os.path.join(root_dir, 'metadata/train/music.csv')
check_files_against_csv(csv_path, files_path, 14)

files_path = os.path.join(root_dir, 'train/speech')
csv_path = os.path.join(root_dir, 'metadata/train/speech.csv')
check_files_against_csv(csv_path, files_path, 5)

files_path = os.path.join(root_dir, 'val/speech')
csv_path = os.path.join(root_dir, 'metadata/val/speech.csv')
check_files_against_csv(csv_path, files_path, 5)

files_path = os.path.join(root_dir, 'test/speech')
csv_path = os.path.join(root_dir, 'metadata/test/speech.csv')
check_files_against_csv(csv_path, files_path, 5)
