import os
from os import listdir
from os.path import isfile, join
from csv import reader
import torchaudio

def check_files_against_csv(csv_path, files_path, index_of_path_in_csv = 7):
    not_missing = []

    with open(csv_path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                path = row[index_of_path_in_csv]
                if os.path.isfile(path):
                    not_missing.append(path.split('/')[3])

    onlyfiles = [f for f in listdir(files_path) if isfile(join(files_path, f))]
    print('Check diff between:', files_path, csv_path)
    diff = list(set(onlyfiles) - set(not_missing))
    print('List of files minus list in csv:', diff)
    diff2 = list(set(not_missing) - set(onlyfiles))
    print('List in csv minus list of files:', diff2)

    print('**** checking stereo track ****')
    mono_files = []
    if index_of_path_in_csv == 7:
        # check if there are any non stereo tracks only in music
        for audio_file_name in onlyfiles:
            audio_path = os.path.join(files_path, audio_file_name)
            audio_metadata = torchaudio.info(audio_path)
            if not audio_metadata.num_channels == 2:
                mono_files.append(audio_path)
        print('Files that are not stereo found in', files_path, ':', mono_files)


files_path = 'podcastmix/test/music'
csv_path = 'podcastmix/metadata/test/music.csv'
check_files_against_csv(csv_path, files_path, 7)

files_path = 'podcastmix/val/music'
csv_path = 'podcastmix/metadata/val/music.csv'
check_files_against_csv(csv_path, files_path, 7)

files_path = 'podcastmix/train/music'
csv_path = 'podcastmix/metadata/train/music.csv'
check_files_against_csv(csv_path, files_path, 7)

files_path = 'podcastmix/train/speech'
csv_path = 'podcastmix/metadata/train/speech.csv'
check_files_against_csv(csv_path, files_path, 5)

files_path = 'podcastmix/val/speech'
csv_path = 'podcastmix/metadata/val/speech.csv'
check_files_against_csv(csv_path, files_path, 5)

files_path = 'podcastmix/test/speech'
csv_path = 'podcastmix/metadata/test/speech.csv'
check_files_against_csv(csv_path, files_path, 5)

