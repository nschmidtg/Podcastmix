# Podcastmix: A dataset for separating music and speech in podcasts

Repository containing the code and precedures to reproduce the [ICASSP publication](TODO) Podcastmix: A dataset for separating music and speech in podcasts.
All links to download the dataset, train, evaluate and separate Podcasts are included here.
Feel free to use the dataset for any other purposes.

## Download the dataset:

If you only want to download the test sets, you can download them from [here](https://zenodo.org/record/5552353)

The train set of the dataset is hosted [here](https://drive.google.com/drive/folders/1tpg9WXkl4L0zU84AwLQjrFqnP-jw1t7z) (~480Gb)

We provide a script to download each of the files quickly, but it requires that you obtain a OAuth2 ApiKey from the Google Developers Console:

- Go to [OAuth 2.0 Playground](https://developers.google.com/oauthplayground/)
- In the “Select the Scope” box, scroll down, expand “Drive API v3”, and select https://www.googleapis.com/auth/drive.readonly
- Click “Authorize APIs” and then “Exchange authorization code for tokens”. Copy the “Access token”.
- Run the following script using the "Access token" as a parameter:
```
/bin/sh download_dataset.sh <Access token>
```

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
The batch number in the ```[MODEL]_model/[MODEL]_config.yml``` file must match the number of GPUs that you want to train with.

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

### Or download the pretrained models

```
wget --no-check-certificate 'https://podcastmix.s3.eu-west-3.amazonaws.com/pretrained_models.zip' -O pretrained_models.zip
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

## Download examples from the synthetic, real and no-reference test sets:

```wget --no-check-certificate 'https://podcastmix.s3.eu-west-3.amazonaws.com/evaluations.zip' -O evaluations.zip```
