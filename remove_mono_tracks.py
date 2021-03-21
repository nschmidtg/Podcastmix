import os
from os import listdir
from os.path import isfile, join
from csv import reader, writer
import torchaudio

def check_files_against_csv(csv_path, files_path, index_of_path_in_csv = 7):
    not_missing = []
    missing = []
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
                else:
                    print(row)
                    missing.append(path)
    print("MISSING!",missing)

    onlyfiles = [f for f in listdir(files_path) if isfile(join(files_path, f))]

    print('****', len(onlyfiles), len(set(onlyfiles)))
    print('****', len(not_missing), len(set(not_missing)))
    print('Check diff between:', files_path, csv_path)
    diff = list(set(onlyfiles) - set(not_missing))
    print('List of files minus list in csv:', diff, len(diff))
    diff2 = list(set(not_missing) - set(onlyfiles))
    print('List in csv minus list of files:', diff2, len(diff2))

def get_non_stereo_files(files_path):
    print('**** checking stereo track ****')
    onlyfiles = [f for f in listdir(files_path) if isfile(join(files_path, f))]
    mono_files = []
    if True:
        # check if there are any non stereo tracks only in music
        for audio_file_name in onlyfiles:
            audio_path = os.path.join(files_path, audio_file_name)
            audio_metadata = torchaudio.info(audio_path)
            if not audio_metadata.num_channels == 2:
                mono_files.append(audio_path)
    print('Files that are not stereo found in', files_path, ':', mono_files)
    return(mono_files)

def create_new_stereo_csv(non_stereo_tracks, csv_path):
    new_csv_path = csv_path.split('.')[0] + '-new.csv'

    print('non_stereo_tracks', non_stereo_tracks)
    with open(new_csv_path, "w") as csv_file:
        with open(csv_path, 'r') as read_obj:
            csv_reader = reader(read_obj)
            # Iterate over each row including the header in the csv
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                path = row[7]
                if not path in non_stereo_tracks:
                    csv_file.write(','.join(row))
                    csv_file.write('\n')
    return new_csv_path

files_path = 'podcastmix/test/music'
csv_path = 'podcastmix/metadata/test/music.csv'

non_stereo_tracks = get_non_stereo_files(files_path)
new_csv_path = create_new_stereo_csv(non_stereo_tracks, csv_path)
check_files_against_csv(new_csv_path, files_path)
