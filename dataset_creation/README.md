## Install dependencies
Create a conda environment:

```
conda env create -f ../environment.yml
```

Activate the environment:

```
conda activate Podcastmix
```

## Download the VCTK dataset:

```
curl https://datashare.ed.ac.uk/download/DS_10283_2651.zip --output VCTK-Corpus.zip && \
unzip -q VCTK-Corpus.zip
```

## Download the music
This script will take the content of ../Jamendo/metadata.json and will download the 19412 songs from the Jamendo music streaming app

```
python download.py
```

#### for MacOS:

```
CERT_PATH=$(python3 -m certifi) && \
export SSL_CERT_FILE=${CERT_PATH} && \
export REQUESTS_CA_BUNDLE=${CERT_PATH} && \
python download.py
```

## Create the dataset
These scripts iterates through the music and speech files and will resample and copy them to the respective train, val or test directories, according to the partition of the set (80%, 10%, 10%). The metadata will be written on csv files.
```
python create_dataset_music.py
python create_dataset_speech.py
```

## Check the podcast consistency
This script will check that the lines contained on the csv metadata files matches the audio files.
```
python check_podcastmix_consistency.py
```
