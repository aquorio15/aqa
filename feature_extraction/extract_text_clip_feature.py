from multilingual_clip import pt_multilingual_clip
import transformers
from PIL import Image
import numpy as np
import clip
import tqdm
import torch
import os
import glob
#%%
save_path = '/nfsshare/Amartya/Pathological_Question_Answering/features/text_embeddings/clip/English/answer/val'
os.makedirs(save_path, exist_ok=True)
# lst = []
with open('/nfsshare/Amartya/Pathological_Question_Answering/multiclass/val.txt', 'r') as f:
    data = f.readlines()
    
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'

device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-L/14", device=device)
# Load Model & Tokenizer
model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

for i, txt in tqdm.tqdm(enumerate(data)):
    text = txt.strip().split('\t')[-2]
    embeddings = model.forward(text, tokenizer)
    with torch.no_grad():
        np.save(f"{save_path}/English_{i+1}.npy", embeddings)
        # image_features = model.encode_image(image)
        # np.save(f"{save_path}/{name}.npy", image_features.cpu().detach().numpy())


# for imgs in tqdm.tqdm(glob.glob('/nfsshare/Amartya/Pathological_Question_Answering/pvqa/images/test/*.jpg')):
#     # print(f"Done: {i}")
#     img = Image.open(imgs)
#     img = img.convert('RGB')
#     name = os.path.basename(imgs).replace('.jpg', '.npy')
#     image = preprocess(img).unsqueeze(0).to(device)
#     # name = file.strip().split('\t')[-2]
#     #name = file.strip().split('\t')[0]
#     # text = file.strip().split('\t')[2]
# with torch.no_grad():
#         # np.save(f"{save_path}/English_{i+1}.npy", embeddings)
#         image_features = model.encode_image(image)
#         np.save(f"{save_path}/{name}.npy", image_features.cpu().detach().numpy())    # embeddings = model.forward(text, tokenizer)
    
    
# %%
