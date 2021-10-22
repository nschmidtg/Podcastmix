import torchaudio
import os

def resample_and_copy(audio_path_dir, destination, destination_sr):
    """
    Checks if the sample_rate is equal to destination_sr. If they are
    different, a resample is done.
    """
    if os.path.isfile(destination):
        return [], True
    audio, original_sr = torchaudio.load(audio_path_dir, normalize=True)
    # if not audio.shape[0] == 2:
    #     return [], True
    # resample from 48kHz -> 44.1kHz
    if not original_sr == destination_sr:
        audio = torchaudio.transforms.Resample(
            original_sr,
            destination_sr
        )(audio)
    torchaudio.save(
        filepath=destination,
        src=audio,
        sample_rate=destination_sr,
        bits_per_sample=16
    )

    return audio, False
