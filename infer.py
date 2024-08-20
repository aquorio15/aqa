from tqdm import tqdm
from datetime import datetime
import time
import os
import collections
import numpy as np
import scipy.io as sio
import torch.utils.data as Data
import torch.optim as optim
import torch.nn as nn
import torch
import scipy
import warnings
import yaml
from conv1d import *
from conv2d import *
from mamba import *
from model import *
from sampler import *
from utils import *
import numpy as np
import torch
import torch.nn.init as init
import torch.nn.functional as F
from typing import List
from calflops import calculate_flops
from dataset import *
from util import *
import shutil
import time
import argparse
import yaml
from sklearn.utils.class_weight import compute_class_weight
import warnings
import subprocess as sp
import nvidia_smi

nvidia_smi.nvmlInit()

handle = nvidia_smi.nvmlDeviceGetHandleByIndex(3)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)


with open("/nfsshare/Amartya/A_question_answering/mamba/config.yaml", "r") as f:
    my_dict = yaml.load(f, Loader=yaml.Loader)

ans2label = pickle.load(
            open(f"/nfsshare/Amartya/A_question_answering/trainval_ans2label.pkl", "rb")
        )

def get_data_tuple(
    splits: str, data_dir: str, lang: str, bs: int, shuffle=False, drop_last=False
) -> DataTuple:
    dset = PVQADataset(splits, data_dir, lang)
    tset = PVQATorchDataset(dset)
    evaluator = PVQAEvaluator(dset)
    data_loader = DataLoader(
        tset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=my_dict["num_workers"],
        drop_last=drop_last,
        pin_memory=True,
        collate_fn=collate_function,
    )
    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

model = MambaFormer(
            patch_size=my_dict["patch_size"],
            stride=my_dict["stride"],
            in_chans=my_dict["in_chans"],
            embed_dim=my_dict["embed_dim"],
            norm_layer=my_dict["norm_layer"],
            flatten=my_dict["flatten"],
            dropout=my_dict["dropout"],
            mode=my_dict["mode"],
            conv_bias=my_dict["conv_bias"],
            drop_rate=my_dict["drop_rate"],
            n_layers=my_dict["n_layers"],
            ape=my_dict["ape"],
            fused_out_size=my_dict["fused_out_size"],
            conv_out=my_dict["conv_out"],
            mid_channels=my_dict["mid_channels"],
            hidden_mlp_size=my_dict["hidden_mlp_size"],
            classifier=len(ans2label)
        )

# model = E2E_MVQA(
#                 input_speech_dim=my_dict["input_speech_dim"],
#                 input_audio_dim=my_dict["input_audio_dim"],
#                 feat_dim=my_dict["feat_dim"],
#                 conv_channels=my_dict["conv_channels"],
#                 d_k=my_dict["d_k"],
#                 d_v=my_dict["d_v"],
#                 d_ff=my_dict["d_ff"],
#                 n_heads=my_dict["n_heads"],
#                 kernel=my_dict["kernel"],
#                 encoder_layers=my_dict["encoder_layers"],
#                 dropout=my_dict["dropout"],
#                 encoder_normalize_before=my_dict["encoder_normalize_before"],
#                 max_seq_len=my_dict["max_seq_len"],
#                 num_classes=self.train_tuple.dataset.num_answers,
#             )

test_tuple = get_data_tuple(
                "test",
                bs=my_dict["valid_batch_size"],
                data_dir=my_dict["audio_dir"],
                lang=my_dict["lang"],
                shuffle=False,
                drop_last=False,
            )

model = model.cuda()
models = torch.load("/nfsshare/Amartya/A_question_answering/mamba/checkpoint_mamba/Hindi_small2/BEST.pth")
# print(state_dict.keys())
model.load_state_dict(models, strict=False)
inputs = {}
dset, loader, evaluator = test_tuple
# audio_speech = torch.ones(1, 16000).cuda()
# audio_sound = torch.ones(1, 16000).cuda()
# inputs['audio_feat'] = audio_speech
# inputs['image'] = audio_sound
start_time = time.time()
model.eval()

# with torch.no_grad():
#     flops, macs, params = calculate_flops(model=model, 
#                                          kwargs = inputs,
#                                          output_precision=4)
#     print("Mamba FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
#     # out = model(audio_speech, audio_sound)
#     print("Used memory:", info.used/1000000)
# end_time = time.time()
# print(f"Time {end_time-start_time}")
# model_dict(input_speech, input_sound)
quesid2ans = {}
for i, datum_tuple in enumerate(loader):
    ques_id, audio_feat1, audio_feat2, target = (datum_tuple)
    target = torch.stack(target)
    with torch.no_grad():
        audio_feat1, audio_feat2, target = (
            audio_feat1.float().cuda(),
            audio_feat2.float().cuda(),
            target.cuda(),
                )
        logit = model(audio_feat1, audio_feat2)
        score, label = logit.max(1)
        for qid, l in zip(ques_id, label.cpu().numpy()):
            ans = dset.label2ans[l]
            quesid2ans[qid] = ans
    evaluator.dump_result(quesid2ans, dump='/DATA/nfsshare/Amartya/A_question_answering/mamba/checkpoint_mamba/Hindi_small/test_result.json')





