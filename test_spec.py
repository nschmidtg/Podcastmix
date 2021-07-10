from PodcastMixSpec import PodcastMixSpec
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
from PodcastMixSpec import PodcastMixSpec
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
    if conf["target_model"] == "UNetSpec":
        sys.path.append('UNetSpec_model')
        AsteroidModelModule = my_import("unet_model.UNet")
    else:
        AsteroidModelModule = my_import("asteroid.models." + conf["target_model"])
    model = AsteroidModelModule.from_pretrained(model_path, sample_rate=conf["sample_rate"])
    print("model_path", model_path)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = PodcastMixSpec(
        csv_dir=conf["test_dir"],
        sample_rate=conf["sample_rate"],
        segment=conf["segment"],
        shuffle_tracks=False,
        multi_speakers=conf["multi_speakers"],
        normalize=False,
        window_size=1024,
        fft_size=1024,
        hop_size=441
    )

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
        # receive magnitude and phases
        mix, sources = test_set[idx]
        # mean and std of magnitude spectrogram
        print("shape of the mix returned from dataloader", mix.shape)
        mean = torch.mean(mix[0])
        std = torch.std(mix[0])
        m_norm = (mix[0] - mean) / std
        mix[0] = m_norm
        # s0 = (sources[0] - torch.mean(mix)) / torch.std(mix)
        # s1 = (sources[1] - torch.mean(mix)) / torch.std(mix)
        m_norm, _ = tensors_to_device([mix, sources], device=model_device)
        est_sources = model(m_norm.unsqueeze(0)).squeeze(0)
        # unnormalize spectrogram
        est_sources = est_sources * std + mean
        
        # pass to cpu
        est_sources = est_sources.cpu()

        # convert spectrograms to audio using mixture phase
        phase = mix[1]
        polar_sources = est_sources * torch.cos(phase) + est_sources * torch.sin(phase) * 1j
        est_sources_audio = torch.istft(polar_sources, 1024, 441, 1024, return_complex=False, onesided=True, center=True)

        # ground truth sources spectrograms to audio
        speech = sources[0]
        music = sources[1]
        polar_speech = speech[0] * torch.cos(speech[1]) + speech[0] * torch.sin(speech[1]) * 1j
        polar_music = music[0] * torch.cos(music[1]) + music[0] * torch.sin(music[1]) * 1j
        polar_mix = mix[0] * torch.cos(phase) + mix[0] * torch.sin(phase) * 1j
        speech_audio = torch.istft(polar_speech, 1024, 441, 1024, return_complex=False, onesided=True, center=True)
        music_audio = torch.istft(polar_music, 1024, 441, 1024, return_complex=False, onesided=True, center=True)
        mix_audio = torch.istft(polar_mix, 1024, 441, 1024, return_complex=False, onesided=True, center=True)

        # remove additional dimention
        speech_out = speech_audio.squeeze(0)
        music_out = music_audio.squeeze(0)
        mix_out = mix_audio.squeeze(0)
        # add both sources to a tensor to return them
        sources = torch.stack([speech_out, music_out], dim=0)

        mix_np = mix_out.cpu().data.numpy()
        sources_np = sources.data.numpy()
        est_sources_np = est_sources_audio.squeeze(0).cpu().data.numpy()
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
            print(mix_np.shape)
            print(sources_np.shape)
            print(est_sources_np.shape)

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
