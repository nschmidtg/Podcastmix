# Note:
This repository is the starting point of the PodcastMix dataset. The main branch is now the forked repo of the [Pompeu Fabra University's Music Technology Group repository](https://github.com/MTG/Podcastmix), where all the most recent updates were uploaded.

# Podcastmix: A dataset for separating music and speech in podcasts

Repository containing the code and precedures to reproduce the [Interspeech 2022 accepted publication "Podcastmix: A dataset for separating music and speech in podcasts"](https://www.isca-archive.org/interspeech_2022/schmidt22_interspeech.html).
All links to download the dataset, train, evaluate and separate Podcasts are included here.
Feel free to use the dataset for any other purposes.

## Pretrained models
If you don't want to train and evaluate the network, but only use the pretrained models, we have uploaded them to a [separated repository](https://github.com/MTG/Podcastmix-inference), so you don't have to download the dataset and could jump right to separate your podcasts.

## Download the dataset:

Download the directory structure with the test sets (podcastmix-real-no-reference and podcastmix-real-with-reference):

```
/bin/bash download_podcastmix.sh
```

This will download the podcastmix-real-no-reference and podcastmix-real-with-reference evaluation partitions. To downloat the podcastmix-synth (synthetically created podcasts from the music and speech files) you have 2 options:

#### Option 1: Download from Google Drive

The train set of the dataset is hosted [here](https://drive.google.com/file/d/1jouTryUzC9u3SNzwHiMN7kjQigXt-PPG/view?usp=sharing) (~480Gb)

We provide a script to download each of the files quickly, but it requires that you obtain a OAuth2 ApiKey from the Google Developers Console:

- Go to [OAuth 2.0 Playground](https://developers.google.com/oauthplayground/)
- In the “Select the Scope” box, scroll down, expand “Drive API v3”, and select `https://www.googleapis.com/auth/drive.readonly`
- Click “Authorize APIs” and then “Exchange authorization code for tokens”. Copy the “Access token”.
- Run the following script using the "Access token" as a parameter:


```
/bin/bash download_podcastmix_synth_set.sh <Access token>
```

---
> **NOTE:**
> The Access Token has an expiration time (1 hour), so if your connection is not fast enough to download all the compressed files within one hour, you will have to refresh the Access Token and re-run the script using the new Access Token. The script will only download the remaining files and not all of them again.

---

#### Option 2: Create the dataset from scratch using the VCTK dataset and the Jamendo/metadata.json file

This option will allow you to recreate the whole dataset starting from the VCTK dataset and the music files specified on the metadata.json. Since it will download the audio, this method does not guarantee that all the audios will still exists either on Jamendo or the VCTK hosting web page.

You will also be able to edit the Jamendo/metadata.json file to, for example, exclude some songs and create a reduced version of the Dataset.

follow [this README](dataset_creation/README.md) to create the podcastmix from scratch.

## Install
Create a conda environment:

```
conda env create -f environment.yml
```

Activate the environment:

```
conda activate Podcastmix
```

## Train network:

```[MODEL]``` could be ```ConvTasNet``` or ```UNet```

### Train

You can specify the GPUs to train, by using CUDA_VISIBLE_DEVICES. 

---
> **NOTE**
> If you want to train the ConvTasNet using 44100 as ```sample_rate```, you will probably need to use at least 2 GPUs due to memory limitations.
---

```
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --config_model [MODEL]_model/[MODEL]_config.yml
```
After each epoch, the system will evaluate the best 10 models so far and save them as checkpoints inside exp/tmp/checkpoints.

### Continue training from checkpoint
If you want to resume the training from a previously saved checkpoint, you can do it using the ```--resume_from``` flag:

```
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --config_model [MODEL]_model/[MODEL]_config.yml \
    --resume_from=<path-to-checkpoint-file>
```
Example:
```
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --config_model UNet_model/UNet_config.yml \
    --resume_from=UNet_model/exp/tmp/checkpoints/epoch\=1-step\=30985.ckpt
```


## Evaluate network:
### Evaluate over the test partition:
``` 
CUDA_VISIBLE_DEVICES=0,1 python test.py --target_model [MODEL] --test_dir podcastmix/podcastmix-synth/metadata/test/ --out_dir=separations --exp_dir=[MODEL]_model/exp/tmp/ --n_save_ex=20 --use_gpu=1
```
### Evaluate over real podcasts with reference:
This script will separate the podcastmix-real-with-reference which consists of the mix and the ground thuth files for music and speech. It will forward the mixes to the network, estimate the separated sources and evaluate the objetive metrics against the ground truth.
```
CUDA_VISIBLE_DEVICES=0,1 python test_real.py --target_model [MODEL] --test_dir podcastmix/podcastmix-real-with-reference/metadata --out_dir=separations --exp_dir=[MODEL]_model/exp/tmp/ --n_save_ex=-1 --use_gpu=1
```

### Notes: ###
- ```--out_dir```: Name of the folder where the separated audios will be saved. The folder will be created inside the ```--exp_dir``` directory.
- ```--n_save_ex```: Integer to indicate the number of examples to be actually saved on the local directory. If -1, then all the examples will be saved.

## Use the model to separate your own podcasts or the real-no-reference set:
You can use your previously trained model or use the [other repository](https://github.com/MTG/Podcastmix-inference) to download pre-trained models and separate them
```
CUDA_VISIBLE_DEVICES=0,1 python forward_podcast.py \
    --test_dir=<directory-of-the-podcastmix-real-no-reference-or-your-own-files> --target_model=[MODEL] \
    --exp_dir=[MODEL]_model/exp/tmp --out_dir=separations \
    --segment=18 --sample_rate=44100 --use_gpu=1
```

### Notes: ###
- ```--segment```: size of the audio to be input to the network and separated in seconds. Due to the size of the convolutions, the UNet only supports 2 + 16*i seconds segments (2, 18, 34, 50, ...). ConvTasNet supports segments of any size.
- ```--out_dir```: Name of the folder where the separated audios will be saved. The folder will be created inside the ```--exp_dir``` directory.
