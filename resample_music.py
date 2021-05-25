import os
import csv
import torchaudio


with open('podcastmix/metadata/test/speech.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            file_path = row[5]
            info = torchaudio.info(file_path)
            print(info.sample_rate, info.bits_per_sample)
            if((not info.sample_rate == 44100) or (not info.bits_per_sample == 16)):
                print(file_path)
                destination = file_path
                audio, original_sf = torchaudio.load(
                     file_path,
                     normalize = True)
                resampled_audio = torchaudio.transforms.Resample(original_sf, 44100)(audio)
                torchaudio.save(filepath=destination, src=resampled_audio, sample_rate=44100, bits_per_sample=16)

