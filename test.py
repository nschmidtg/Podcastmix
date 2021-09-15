import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from utils.my_import import my_import

from asteroid.metrics import get_metrics
from pytorch_lightning import seed_everything
from PodcastMixMulti import PodcastMixMulti
from asteroid.metrics import MockWERTracker

seed_everything(1, workers=True)



parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_dir",
    type=str,
    required=True,
    help="Test directory including the csv files"
)
parser.add_argument(
    "--target_model",
    type=str,
    required=True,
    help="Asteroid model to use"
)
parser.add_argument(
    "--out_dir",
    type=str,
    default='ConvTasNet/eval/tmp',
    required=True,
    help="Directory where the eval results" " will be stored",
)
parser.add_argument(
    "--use_gpu",
    type=int,
    default=0,
    help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir",
                    default="exp/tmp",
                    help="Best serialized model path")
parser.add_argument(
    "--n_save_ex",
    type=int,
    default=10,
    help="Number of audio examples to save, -1 means all"
)
parser.add_argument(
    "--compute_wer",
    type=int,
    default=0,
    help="Compute WER using ESPNet's pretrained model"
)

COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]

def main(conf):
    compute_metrics = COMPUTE_METRICS
    wer_tracker = (
        MockWERTracker()
    )
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    if conf["target_model"] == "UNet":
        sys.path.append('UNet_model')
        AsteroidModelModule = my_import("unet_model.UNet")
    else:
        sys.path.append('ConvTasNet_model')
        AsteroidModelModule = my_import("conv_tasnet_norm.ConvTasNetNorm")
    model = AsteroidModelModule.from_pretrained(model_path, sample_rate=conf["sample_rate"])
    # model = ConvTasNet
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    test_set = PodcastMixMulti(
        csv_dir=conf["test_dir"],
        sample_rate=conf["sample_rate"],
        original_sample_rate=conf["original_sample_rate"],
        segment=conf["segment"],
        shuffle_tracks=False,
        multi_speakers=conf["multi_speakers"]
    )
    # Used to reorder sources only

    # Randomly choose the indexes of sentences to save.
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    ex_save_dir = os.path.join(eval_save_dir, "examples/")
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    series_list = []

    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources = test_set[idx]

        if conf["target_model"] == "UNet":
            mix = mix.unsqueeze(0)
        # get audio representations, pass the mix to the unet, it will normalize
        # it, create the masks, pass them to audio, unnormalize them and return
        est_sources = model(mix)

        mix_np = mix.cpu().data.numpy()
        if conf["target_model"] == "UNet":
            mix_np = mix_np.squeeze(0)

        sources_np = sources.cpu().data.numpy()
        est_sources_np = est_sources.squeeze(0).cpu().data.numpy()

        try:
            utt_metrics = get_metrics(
                mix_np,
                sources_np,
                est_sources_np,
                sample_rate=conf["sample_rate"],
                metrics_list=COMPUTE_METRICS,
                average=False
            )
            series_list.append(pd.Series(utt_metrics))
        except:
            print("Error. Index", idx)
            print(mix_np)
            print(sources_np)
            print(est_sources_np)

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx + 1))
            os.makedirs(local_save_dir, exist_ok=True)
            print(mix_np.shape)
            sf.write(
                local_save_dir + "mixture.wav",
                mix_np,
                conf["sample_rate"]
            )
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(
                    local_save_dir + "s{}.wav".format(src_idx),
                    src,
                    conf["sample_rate"]
                )
            for src_idx, est_src in enumerate(est_sources_np):
                # est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx),
                    est_src,
                    conf["sample_rate"],
                )
        # Write local metrics to the example folder.
        with open(local_save_dir + "metrics.json", "w") as f:
            json.dump({k:v.tolist() for k,v in utt_metrics.items()}, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()

    print("Overall metrics :")
    print(final_results)
    if conf["compute_wer"]:
        print("\nWER report")
        wer_card = wer_tracker.final_report_as_markdown()
        print(wer_card)
        # Save the report
        with open(os.path.join(eval_save_dir, "final_wer.md"), "w") as f:
            f.write(wer_card)

    with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
        json.dump({k:v.tolist() for k,v in final_results.items()}, f, indent=0)

    # for publishing the model:
    # model_dict = torch.load(model_path, map_location="cpu")
    # os.makedirs(os.path.join(conf["exp_dir"], "publish_dir"), exist_ok=True)
    # publishable = save_publishable(
    #     os.path.join(conf["exp_dir"], "publish_dir"),
    #     model_dict,
    #     metrics=final_results,
    #     train_conf=train_conf,
    # )


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["segment"] = train_conf["data"]["segment"]
    arg_dic["original_sample_rate"] = train_conf["data"]["original_sample_rate"]
    arg_dic["multi_speakers"] = train_conf["training"]["multi_speakers"]
    arg_dic["train_conf"] = train_conf

    main(arg_dic)
