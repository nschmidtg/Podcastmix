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
from torch.utils.data.dataset import Dataset
import torchaudio
import sys

from asteroid.metrics import get_metrics
from pytorch_lightning import seed_everything
from asteroid.utils import tensors_to_device
from asteroid.metrics import MockWERTracker

seed_everything(1)

class PodcastLoader(Dataset):
    dataset_name = "PodcastMix"
    def __init__(self, csv_dir, sample_rate=44100):
        self.csv_dir = csv_dir
        self.sample_rate = sample_rate
        self.mix_csv_path = os.path.join(self.csv_dir, 'mix.csv')
        self.df_mix = pd.read_csv(self.mix_csv_path, engine='python')
        torchaudio.set_audio_backend(backend='soundfile')

    def __len__(self):
        return len(self.mix_csv_path)
    
    def __getitem__(self, index):
        row = self.df_mix.iloc[index]
        podcast_path = row['mix_path']
        speech_path = row['speech_path']
        music_path = row['music_path']
        sources_list = []
        # breakpoint()
        mixture, _ = torchaudio.load(
            podcast_path
        )
        speech, _ = torchaudio.load(
            speech_path
        )
        music, _ = torchaudio.load(
            music_path
        )
        sources_list.append(speech)
        sources_list.append(music)
        sources = np.vstack(sources_list)
        sources = torch.from_numpy(sources)

        return mixture, sources



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

COMPUTE_METRICS = ["sdr", "sir", "sar"]


def my_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


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
        AsteroidModelModule = my_import("asteroid.models." + conf["target_model"])
    model = AsteroidModelModule.from_pretrained(model_path, sample_rate=conf["sample_rate"])
    print("model_path", model_path)
    # model = ConvTasNet
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = PodcastLoader(
        conf["test_dir"],
    )  # Uses all segment length
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
        m_norm = (mix - torch.mean(mix)) / torch.std(mix)
        # s0 = (sources[0] - torch.mean(mix)) / torch.std(mix)
        # s1 = (sources[1] - torch.mean(mix)) / torch.std(mix)
        m_norm, _ = tensors_to_device([m_norm, sources], device=model_device)
        if conf["target_model"] == "UNet":
            est_sources = model(m_norm.unsqueeze(0)).squeeze(0)
        else:
            est_sources = model(m_norm)
        # unnormalize
        est_sources = est_sources * torch.std(mix) + torch.mean(mix)

        mix_np = mix.cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = est_sources.squeeze(0).cpu().data.numpy()
        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics
        try:
            utt_metrics = get_metrics(
                mix_np,
                sources_np,
                est_sources_np,
                sample_rate=conf["sample_rate"],
                metrics_list=COMPUTE_METRICS,
            )
            series_list.append(pd.Series(utt_metrics))
        except:
            print("Error. Index", idx)
            print(mix_np)
            print(sources_np)
            print(est_sources_np)

        # Save some examples in a folder. Wav files and metrics as text.
        
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(
                local_save_dir + "mixture.wav",
                mix_np[0],
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
                est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx),
                    est_src,
                    conf["sample_rate"],
                )
            # Write local metrics to the example folder.
            with open(local_save_dir + "metrics.json", "w") as f:
                json.dump(utt_metrics, f, indent=0)

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
        json.dump(final_results, f, indent=0)

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
    arg_dic["multi_speakers"] = train_conf["training"]["multi_speakers"]
    arg_dic["train_conf"] = train_conf

    main(arg_dic)


"""
usage:
python test_real.py --target_model ConvTasNet --test_dir podcastmix/test-real/metadata --out_dir=ConvTasNet/eval/tmp --exp_dir=../../Desktop/experiments-defense-epochs/ConvTasNet_model/exp-92-epochs-LogL1/tmp/ --use_gpu=0
"""
