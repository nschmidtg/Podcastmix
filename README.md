# Master Thesis

## Installation
Create a conda environment:
```conda create --name thesis python=3.7```

```conda activate thesis```

```pip install -r requirements.txt```

Download the dataset:

```wget --no-check-certificate 'https://podcastmix.s3.eu-west-3.amazonaws.com/podcastmix.zip' -O podcastmix.zip```

Unzip it:

```unzip podcastmix.zip```

## Train network:
The batch number in the ```[MODEL]_model/[MODEL]_config.yml``` file must match the number of GPUs that you want to train with.

```[MODEL]``` can be any of the following:

- ConvTasNet
- UNet

### Train
```
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --config_model [MODEL]_model/[MODEL]_config.yml
```

### Or download the pretrained models

```wget --no-check-certificate 'https://podcastmix.s3.eu-west-3.amazonaws.com/pretrained_models.zip' -O pretrained_models.zip```

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
    --out_dir=<where-to-save-separations> --exp_dir=<path to best_model.pth>
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
    --segment=18 --sample_rate=44100

## Download examples from the synthetic, real and no-reference test sets:

```wget --no-check-certificate 'https://podcastmix.s3.eu-west-3.amazonaws.com/evaluations.zip' -O evaluations.zip```
