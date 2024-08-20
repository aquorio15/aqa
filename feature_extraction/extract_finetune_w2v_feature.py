import librosa
import torch
import tqdm
import os
import glob
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
save_path = '/DATA/nfsshare/Amartya/Audio_question_answering/audio_features/w2v'
model_name = "/DATA/nfsshare/Amartya/Audio_question_answering/code/wav2vec2-xlsr-audio-classification/checkpoint-3800"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
model = model.cuda()
os.makedirs(save_path, exist_ok=True)

audio_path = glob.glob('/DATA/nfsshare/Amartya/Audio_question_answering/audio_files/*.wav')
for path in tqdm.tqdm(audio_path):
    name = os.path.basename(path).replace('.wav', '.npy')
    input_audio, sample_rate = librosa.load(path, sr=16000)
    i= feature_extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate)
    with torch.no_grad():
        o = model(i.input_values.cuda())
        np.save(f"{save_path}/{name}", o.extract_features.cpu().numpy())
