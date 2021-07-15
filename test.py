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

from asteroid.metrics import get_metrics
from pytorch_lightning import seed_everything
from PodcastMixMulti import PodcastMixMulti
from asteroid.utils import tensors_to_device
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


def my_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def compute_mag_phase(torch_signals, fft_size, hop_size, window):
    X_in = torch.stft(torch_signals, fft_size=fft_size, hop_size=hop_size, window=window, return_complex=False, normalized=True)
    real, imag = X_in.unbind(-1)
    complex_n = torch.cat((real.unsqueeze(1), imag.unsqueeze(1)), dim=1).permute(0,2,3,1).contiguous()
    r_i = torch.view_as_complex(complex_n)
    phase = torch.angle(r_i)
    X_in = torch.sqrt(real**2 + imag**2)
    # concat mag and phase: [torch_signals, mag/phase, n_bins, n_frames]
    torch_signals = torch.cat((X_in.unsqueeze(1), phase.unsqueeze(1)), dim=1)
    return torch_signals


def main(conf):
    compute_metrics = COMPUTE_METRICS
    wer_tracker = (
        MockWERTracker()
    )
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    if conf["target_model"] == "UNetSpec":
        sys.path.append('UNetSpec_model')
        AsteroidModelModule = my_import("unet_model.UNet")
    else:
        AsteroidModelModule = my_import("asteroid.models." + conf["target_model"])
    model = AsteroidModelModule.from_pretrained(model_path, sample_rate=conf["sample_rate"])
    # model = ConvTasNet
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = PodcastMixMulti(
        csv_dir=conf["test_dir"],
        sample_rate=conf["sample_rate"],
        original_sample_rate=["original_sample_rate"],
        segment=conf["segment"],
        domain='time',
        fft_size=conf["fft_size"],
        window_size=conf["window_size"],
        hop_size=conf["hop_size"],
        shuffle_tracks=False,
        multi_speakers=conf["multi_speakers"],
        normalize=False
    )
    # Used to reorder sources only

    # Randomly choose the indexes of sentences to save.
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    ex_save_dir = os.path.join(eval_save_dir, "examples/")
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    series_list = []

    # # read mean and std from json
    with open('mean_std.json') as json_file:
        data = json.load(json_file)
        mean = data['sum_accum_mean'] / data['n_items']
        std = data['sum_accum_std'] / data['n_items']
    window = torch.hamming_window(conf["window_size"])

    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources = test_set[idx]
        if conf["target_model"] == "UNetSpec":
            # get audio from dataloader, normalize mix, pass to spectrogram
            # forward spectrogram to model, transform spectrograms to audio
            # using mix phase, unnormalize estimated sources and
            # compare them with the ground truth sources 
            mix_audio_norm = (mix - mean) / std
            
            # audio to spectrogram
            mix_audio_norm = mix_audio_norm.unsqueeze(0)
            mix_norm = compute_mag_phase(mix_audio_norm, conf["fft_size"], conf["hop_size"], window=window)
            mix_norm = mix_audio_norm.squeeze(0)
            
            sources = compute_mag_phase(sources, conf["fft_size"], conf["hop_size"], window=window)
            m_norm, _ = tensors_to_device([mix_norm, sources], device=model_device)
            est_sources = model(m_norm.unsqueeze(0)).squeeze(0)
            
            # pass to cpu
            est_sources = est_sources.cpu()

            # convert spectrograms to audio using mixture phase
            polar_sources = est_sources * torch.cos(mix_norm[1]) + est_sources * torch.sin(mix_norm[1]) * 1j
            est_sources_audio = torch.istft(polar_sources, conf["fft_size"], conf["hop_size"], window=window, return_complex=False, onesided=True, center=True, normalized=True)

            # ground truth sources spectrograms to audio
            speech = sources[0]
            music = sources[1]
            
            # unnormalize estimated sources:
            est_sources_audio = est_sources_audio * std + mean

            # remove additional dimention
            speech_out = speech.squeeze(0)
            music_out = music.squeeze(0)
            mix_out = mix.squeeze(0)
            # add both sources to a tensor to return them
            sources = torch.stack([speech_out, music_out], dim=0)

            mix_np = mix_out.cpu().data.numpy()
            sources_np = sources.data.numpy()
            est_sources_np = est_sources_audio.squeeze(0).cpu().data.numpy()        
        else:
            # get audio representations, normalize it, forward to convtasnet
            # unnormalize estimated sources and compare them with the
            # ground truth
            m_norm = (mix - mean) / std
            m_norm, _ = tensors_to_device([m_norm, sources], device=model_device)
            est_sources = model(m_norm)
            # unnormalize
            est_sources = est_sources * std + mean

            mix_np = mix.cpu().data.numpy()
            sources_np = sources.cpu().data.numpy()
            est_sources_np = est_sources.squeeze(0).cpu().data.numpy()
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
CUDA_VISIBLE_DEVICES=1 python test.py --target_model ConvTasNet \
    --test_dir augmented_dataset/metadata/test/ --task linear_mono \
        --out_dir=ConvTasNet_model/eval/tmp --exp_dir=ConvTasNet_model/exp/tmp
"""
