from huggingface_whisper_feature_extractor import HuggingFaceWhisper
import speechbrain as sb
import glob
import os
import tqdm
import torch
import numpy as np
save_path = '/DATA/nfsshare/Amartya/Audio_question_answering/audio_features/whisper/'
os.makedirs(save_path, exist_ok=True)
model_hub_whisper='/DATA/nfsshare/Amartya/Audio_question_answering/code/whisper-large-v3-audio-classification/checkpoint-36850'
model_whisper = HuggingFaceWhisper(model_hub_whisper)
model_whisper = model_whisper.cuda()
model_whisper.encoder_only=True
source_path = glob.glob('/DATA/nfsshare/Amartya/Audio_question_answering/audio_files/*.wav')
for path in tqdm.tqdm(source_path):
    name = os.path.basename(path).replace('.wav', '.npy')
    source = sb.dataio.dataio.read_audio(path).squeeze()
    source = source.unsqueeze(0)
    with torch.no_grad():
        fea_whisper = model_whisper(source.cuda())
    np.save(f"{save_path}/{name}",fea_whisper.cpu().numpy())
