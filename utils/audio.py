import numpy as np
import torch
import os
import subprocess
import torchaudio
from tempfile import NamedTemporaryFile
import torch

def load_audio(path):
    sound, _ = torchaudio.load(path)
    sound = sound / torch.max(torch.abs(sound))  # Normalize manually if needed

    sound = sound.numpy().T
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound

def get_audio_length(path):
    output = subprocess.check_output(
        ['soxi -D \"%s\"' % path.strip()], shell=True)
    return float(output)

def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                               tar_filename, start_time,
                                                                                               end_time)
        os.system(sox_params)
        y = load_audio(tar_filename)
        return y

def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(
            tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(
            path, sample_rate, augmented_filename, " ".join(sox_augment_params))
        os.system(sox_params)
        y = load_audio(augmented_filename)
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15), gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio