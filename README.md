# Master Thesis

## Installation
Create a conda environment:
```conda create --name thesis python=3.7```

```conda activate thesis```

```sh install_dependencies.sh```

Download the dataset:

```wget --no-check-certificate 'https://podcastmix.s3.eu-west-3.amazonaws.com/podcastmix.zip' -O podcastmix.zip```

Unzip it:

```unzip podcastmix.zip```

## Train network:
The batch number in the ```[MODEL]_model/[MODEL]_config.yml``` file must match the number of GPUs that you want to train with.

```[MODEL]``` can be any of the following:

- ConvTasNet
- DPTNet
- UNet
- SuDORMRFNet
- LSTMTasNet
- DPRNNTasNet

### Train
```
CUDA_VISIBLE_DEVICES=1 python train.py --config_model [MODEL]_model/[MODEL]_config.yml
```

### Continue training from checkpoint
```
CUDA_VISIBLE_DEVICES=1 python train.py --config_model [MODEL]_model/[MODEL]_config.yml --resume_from= [MODEL]_model/exp/tmp/checkpoints/epoch\=28-step\=224691.ckpt
```


## Evaluate network:

``` 
CUDA_VISIBLE_DEVICES=1 python test.py --target_model [MODEL] \
    --test_dir augmented_dataset/metadata/test/ --task linear_mono \
        --out_dir=[MODEL]_model/eval/tmp --exp_dir=[MODEL]_model/exp/tmp
```