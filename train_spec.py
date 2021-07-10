import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DDPPlugin
import sys

from PodcastMixSpec import PodcastMixSpec
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
#from l2 import L2Time
from logl2 import LogL2Spec
#from torch.nn import L1Loss
seed_everything(1, workers=True)

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir",
    default="exp/tmp",
    help="Full path to save best validation model"
)

def main(conf):
    train_set = PodcastMixSpec(
        csv_dir=conf["data"]["train_dir"],
        sample_rate=conf["data"]["sample_rate"],
        segment=conf["data"]["segment"],
        shuffle_tracks=True,
        multi_speakers=conf["training"]["multi_speakers"],
        normalize=conf["training"]["normalize"],
        window_size=1024,
        fft_size=1024,
        hop_size=441,
    )

    val_set = PodcastMixSpec(
        csv_dir=conf["data"]["valid_dir"],
        sample_rate=conf["data"]["sample_rate"],
        segment=conf["data"]["segment"],
        shuffle_tracks=True,
        multi_speakers=conf["training"]["multi_speakers"],
        normalize=conf["training"]["normalize"],
        window_size=1024,
        fft_size=1024,
        hop_size=441,
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        pin_memory=True
    )
    if(conf["model"]["name"] == "UNetSpec"):
        sys.path.append('UNetSpec_model')
        from unet_model import UNet
        model = UNet(
            conf["data"]["sample_rate"],
            # conf["stft"]["fft_size"],
            # conf["stft"]["hop_size"],
            # conf["stft"]["window_size"],
            conf["convolution"]["kernel_size"],
            conf["convolution"]["stride"]
        )
        optimizer = make_optimizer(model.parameters(), **conf["optim"])
        if conf["training"]["half_lr"]:
            scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                factor=0.5,
                patience=5
            )

    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["model"]["name"] + "_model/" + conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    loss_func = LogL2Spec()
    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=50,
            verbose=True
        ))

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend=distributed_backend,
        # limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        resume_from_checkpoint=conf["main_args"]["resume_from"],
        precision=32,
        plugins=DDPPlugin(find_unused_parameters=False)
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        print(best_k,f)
        json.dump(best_k, f, indent=0)
    print(checkpoint.best_model_path)
    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    import sys
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    parser.add_argument(
        "--config_model", type=str, required=True, help="Asteroid model to use"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="path to the desired restore checkpoint with .ckpt extension"
    )
    config_model = sys.argv[2]
    with open(config_model) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    main(arg_dic)

"""
usage:
CUDA_VISIBLE_DEVICES=1 python train.py --config_model \
    ConvTasNet_model/ConvTasNet_config.yml \
       --resume_from=DPTNet_model/exp/tmp/checkpoints/epoch\=28-step\=224691.ckpt
"""