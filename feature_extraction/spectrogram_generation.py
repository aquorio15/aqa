import numpy as np # Import Numpy Package 
import librosa # Import librosa package to read and plot sound spectrogram
import librosa.display # Import librosa package to display spectrogram
import matplotlib.pyplot as plt # Import matplotlib for data visualization
import pandas as pd # Import pandas 
import os # import pandas to make use of listdir
import h5py # import H5py package to create database from the given signals
import soundfile as sf # import soundfile to read channels of a signal
from scipy import signal # import scipy 
import math 
from PIL import Image, ImageOps
import wave, math, array, argparse, sys, timeit
#from colorspace import diverging_hcl, desaturate,deutan, protan
# def Normalization(x):
#     x_norm= (x-np.min(x))/(np.max(x)-np.min(x))
    
#     return x_norm
# def DNormalization(x):
#     x_dnorm= (x* (np.max(x)-np.min(x)))+np.min(x)
#     return x_dnorm 
# audio,fs = sf.read("Thunder.wav", dtype='float32')
# audio=Normalization(audio)
#print(np.shape(audio))
# n_fft = 2048
# hop_length = 1024
# n_mels=128
# fmax=fs//2
def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
  wav,sr = librosa.load(file_path,sr=sr)
  TL=5
  if wav.shape[0]<TL*sr:
    wav=np.pad(wav,int(np.ceil((TL*sr-wav.shape[0])/2)),mode='reflect')
  else:
    wav=wav[:TL*sr]
  spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
              hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
  spec_db=librosa.power_to_db(spec,top_db=top_db)
  return spec_db
def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  print(np.shape(spec_scaled))
  return spec_scaled
filename='/nfsshare/Amartya/A_question_answering/Audio/English/train/English_1.wav' 
SG_Image=spec_to_image(get_melspectrogram_db(filename))
#print(SG_Image)
#print(np.shape(SG_Image))
#plt.imshow(SG_Image)
# # D = np.abs(librosa.stft(audio, n_fft=n_fft,  hop_length=hop_length))
Signal,fs = librosa.load(filename)
librosa.display.specshow(SG_Image, sr=fs,y_axis='linear',cmap='coolwarm');
#librosa.display.specshow(SG_Image, sr=fs,cmap='coolwarm');
#plt.figure(figsize=(30,14))
#plt.imshow(SG_Image,cmap='coolwarm')
plt.colorbar(orientation="horizontal", pad=0.1)
plt.ylabel("Frequency",size=15)
plt.xlabel("Time",size=15)
#plt.title('Baby Crying')
plt.xticks(size=25)
plt.savefig("/nfsshare/Amartya/A_question_answering/Audio/English_1.pdf") 
plt.show()
#a=a[torch.randperm(a.size()[0])]
# # print(np.shape(D))
# stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,win_length=None, window='hann',center=True, pad_mode='reflect')
# stft = np.abs(stft)
# #freqs = librosa.core.fft_frequencies(sr=fs, n_fft=n_fft)
# #stft = librosa.perceptual_weighting(stft**2, freqs, ref=1.0, amin=1e-10, top_db=99.0)
# D = librosa.feature.melspectrogram(S=stft, sr=fs, n_mels=n_mels, fmax=fmax,n_fft=n_fft)
# np.save('Thunder',D[:,:-2])
# print(np.shape(D[:,:-2]))
# plt.specgram(audio,Fs=fs) 
# #D=stft
# #print(np.shape(D))
# #D=np.exp(np.log(D+1e-10))
# #print("D=\n",D)
# #librosa.display.specshow(D, sr=fs, x_axis='time', y_axis='linear');
# #plt.colorbar();
# plt.show()
# # res= librosa.feature.inverse.mel_to_audio(D,hop_length=hop_length, sr=fs,n_fft=n_fft,fmax=fmax,win_length=None, window='hann',center=True, pad_mode='reflect')
# # print(np.shape(audio))
# # print("Audio=\n",audio)
# # print("Norm Audio=\n",Normalization(audio))
# # print("Achyut\n")
# # print("R Audio=\n",Normalization(res))
# # plt.plot(Normalization(res))
# # #plt.show()


# # librosa.output.write_wav('O_Audio.wav', Normalization(audio), fs)
# # librosa.output.write_wav('R_Audio.wav', Normalization(res), fs)
     

