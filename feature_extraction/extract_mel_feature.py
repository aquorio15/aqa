from pathlib import Path

import os
import numpy as np

import torch
from torch import nn
import whisper
import torchaudio
import torchaudio.transforms as at
from natsort import natsorted
import fastnumpyload
import tqdm
import glob
#paths = '/nfsshare/Amartya/Pathological_Question_Answering/features/mel/German/test' 
#os.makedirs(paths, exist_ok=True)

def exact_div(x, y):
    assert x % y == 0
    return x // y

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 5
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE 
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)

def load_wave(wave_path, sample_rate: int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array

for path in tqdm.tqdm(natsorted(glob.glob('/DATA/nfsshare/Amartya/Audio_question_answering/Audio/*/*/*.wav'))):
    audio = load_wave(path, sample_rate=SAMPLE_RATE)
    new_path = os.path.dirname(path).replace('/Audio/', '/features/mel/')
    os.makedirs(new_path, exist_ok=True)
    filename = os.path.basename(path).replace('.wav', '.npy')
    #dirname = os.path.dirname(path)
    if not os.path.exists(f'{new_path}/{filename}'):
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, n_mels=80).unsqueeze(0)
        np.save(f'{new_path}/{filename}', mel)
        torch.cuda.empty_cache()
        del mel