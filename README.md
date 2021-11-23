# Podcastmix: A dataset for separating music and speech in podcasts

Repository containing the code and precedures to reproduce the ICASSP 2022 submission "Podcastmix: A dataset for separating music and speech in podcasts".
All links to download the dataset, train, evaluate and separate Podcasts are included here.
Feel free to use the dataset for any other purposes.

## Pretrained models
If you don't want to train and evaluate the network, but only use the pretrained models, we have uploaded them to a [separated repository](https://github.com/MTG/Podcastmix-inference), so you don't have to download the dataset and could jump right to separate your podcasts.

## Download the dataset:

Download the directory structure with the test sets (podcastmix-real-no-reference and podcastmix-real-with-reference):

```
/bin/bash download_podcastmix.sh
```

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

```
[MODEL]
```

can be any of the following:

- ConvTasNet
- UNet

### Train

```
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --config_model [MODEL]_model/[MODEL]_config.yml
```

### Continue training from checkpoint

```
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --config_model [MODEL]_model/[MODEL]_config.yml \
    --resume_from=<path-to-checkout-file>
```


## Evaluate network:
### Evaluate over the test partition:
``` 
CUDA_VISIBLE_DEVICES=0,1 python test.py --target_model [MODEL] \
    --test_dir podcastmix/metadata/test/ \
    --out_dir=<where-to-save-separations> --exp_dir=<path to best_model.pth> --use_gpu=1
```
### Evaluate over real podcasts:
```
CUDA_VISIBLE_DEVICES=0,1 python test_real.py --target_model [MODEL] \
    --test_dir podcastmix/real_podcasts/metadata --out_dir=<where-to-save-separations> \
    --exp_dir=<path to best_model.pth> --use_gpu=1 --n_save_ex=-1
```

## Use the model to separate podcasts:
```
CUDA_VISIBLE_DEVICES=0,1 python forward_podcast.py \
    --test_dir=podcastsmix/podcasts_no_reference --target_model=[MODEL] \
    --exp_dir=<path to best_model.pth> --out_dir=<where-to-save-separations> \
    --segment=18 --sample_rate=44100 --use_gpu=1
```
