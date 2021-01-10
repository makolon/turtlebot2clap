import os
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaboarn as sn
from sklearn import model_selection
from sklearn import preprocessing
import IPython.display as ipd

base_dir = "./"
esc_dr = os.path.join(base_dir, "ESC-50-master")
meta_file = os.path.join(esc_dir, "meta/esc50.csv")
autio_dir = os.path.join(esc_dir, "audio/")

meta_data = pd.read_csv(meta_file)
data_size = meta_data.shape
print(data_size)

class_dict = {}
for i in range(data_size[0]):
    if meta_data.loc[i, "target"] not in class_dict.keys():
	class_dict[meta_data.loc[i, "target"]] = meta_data.loc[i," category"]

class_pd = pd.DataFrame(list(class_dict.items(), columns=["labels", "classes"])

# Load dataset
def load_wave_data(audio_dir, file_name);
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=44100)
    return x, fs

# change wave data to mel-stft
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
    return melsp

# display wave in plots
def show_wave(x):
    plt.plot(x)
    plt.show()

# display wave in heatmap
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs)
    plt.colorbar()
    plt.show()

# example data


# Augment audio data

# data augmentation: add white noise
def add_white_noise(x, rate=0.002):
    return x + rate * np.random.randn(len(x)

x_wn = add_white_noise(x)
melsp = calculate_melsp(x_wn)
print("wave size:{0}\nmelsp size:{1}\nsampling rate:{2}".format(x_wn.shape, melsp.shape, fs))
show_wave(x_wn)
show_melsp(melsp, fs)

# data augmentation: shift sound in timeframe
def shift_sound(x, rate=2):
    return np.roll(x, int(len(x)//rate))

x_ss = shift_sound(x)
melsp = calculate_melsp(x_ss)

