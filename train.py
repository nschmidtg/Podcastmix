import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from PodcastMix import PodcastMix
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

import importlib

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")

def main(conf):
    train_set = PodcastMix(
        csv_dir=conf["data"]["train_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
    )

    val_set = PodcastMix(
        csv_dir=conf["data"]["valid_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    # print(conf)
    optimizer = None

    if(conf["model"]["name"] == "ConvTasNet"):
        from asteroid.models import ConvTasNet

        conf["masknet"].update({"n_src": conf["data"]["n_src"]})
        # Define scheduler
        scheduler = None
        if conf["training"]["half_lr"]:
            scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
        model = ConvTasNet(
            **conf["filterbank"], 
            **conf["masknet"], 
            sample_rate=conf["data"]["sample_rate"]
        )
    elif(conf["model"]["name"] == "DCCRNet"):
        # Not working
        from asteroid.models import DCCRNet

        model = DCCRNet(
            sample_rate=conf["data"]["sample_rate"], 
            architecture=conf["model"]["architecture"],
        )
    elif(conf["model"]["name"] == "DPRNNTasNet"):
        from asteroid.models import DPRNNTasNet

        # CHECK! Define scheduler
        scheduler = None
        if conf["training"]["half_lr"]:
            scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
        model = DPRNNTasNet(
            n_src=conf["data"]["n_src"],
            sample_rate=conf["data"]["sample_rate"],
            **conf["model_init"]
        )
    elif(conf["model"]["name"] == "DPTNet"):
        from asteroid.models import DPTNet
        print("hola!")
        conf["masknet"].update({"n_src": train_set.n_src})
        model = DPTNet(
            sample_rate=conf["data"]["sample_rate"],
            **conf["filterbank"],
            **conf["masknet"]
        )
        optimizer = make_optimizer(model.parameters(), **conf["optim"])
        print("chao")
        from asteroid.engine.schedulers import DPTNetScheduler

        scheduler = {
            "scheduler": DPTNetScheduler(
                optimizer, len(train_loader) // conf["training"]["batch_size"], 64
            ),
            "interval": "step",
        }

    elif(conf["model"]["name"] == "DeMask"):
        from asteroid.models import DeMask
        model = DeMask(
            sample_rate=conf["data"]["sample_rate"],
            **conf["model_init"]
        )
    elif(conf["model"]["name"] == "DCUNet"):
        from asteroid.models import DCUNet
        model = DCUNet(
            architecture=conf["model"]["architecture"]
        )
    elif(conf["model"]["name"] == "LSTMTasNet"):
        from asteroid.models import LSTMTasNet

        # CHECK! Define scheduler
        scheduler = None
        if conf["training"]["half_lr"]:
            scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
        model = LSTMTasNet(
            n_src=conf["data"]["n_src"],
            sample_rate=conf["data"]["sample_rate"],
            **conf["model_init"]
        )
    elif(conf["model"]["name"] == "SuDORMRFNet"):
        from asteroid.models import SuDORMRFNet

        # CHECK! Define scheduler
        scheduler = None
        if conf["training"]["half_lr"]:
            scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
        model = SuDORMRFNet(
            n_src=conf["data"]["n_src"],
            sample_rate=conf["data"]["sample_rate"],
            **conf["model_init"]
        )
    if(optimizer == None):
        optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Just after instantiating, save the args. Easy loading in the future.
    # exp_dir = conf["main_args"]["exp_dir"]
    exp_dir = conf["model"]["name"] + "_model/" + conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend=distributed_backend,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

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
    config_model = sys.argv[2]
    with open(config_model) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    print(parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)

"""
usage: 
CUDA_VISIBLE_DEVICES=1 python train.py --config_model ConvTasNet_model/ConvTasNet_config.yml
"""
