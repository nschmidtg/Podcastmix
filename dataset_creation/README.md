## Download the VCTK dataset:
```
curl https://datashare.ed.ac.uk/download/DS_10283_2651.zip --output vctk.zip
unzip vctk.zip .
```

## Download the music

```
python3 download.py
```

### on MacOS:

```
CERT_PATH=$(python3 -m certifi) && \
export SSL_CERT_FILE=${CERT_PATH} && \
export REQUESTS_CA_BUNDLE=${CERT_PATH} && \
python3 download.py
```

## create the dataset

```
python create_dataset_music.py
python create_dataset_speech.py
```

## check the podcast consistency
```
python check_podcastmix_consistency.py
```
