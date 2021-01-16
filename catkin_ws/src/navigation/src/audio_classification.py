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

class Audio_Classifer(object):
    def __init__(self):
        self.meta_data = pd.read_csv(meta_file)
        self.data_size = meta_data.shape

    def arrange_target(self):
        class_dict = {}
        for i in range(self.data_size[0]):
            if self.meta_data.loc[i, "target"] not in class_dict.keys():
                class_dict[self.meta_data.loc[i, "target"]] = self.meta_data.loc[i," category"]

        class_pd = pd.DataFrame(list(class_dict.items()), columns=["labels", "classes"])
        return class_pd

    # Load dataset
    def load_wave_data(self, audio_dir, file_name):
        file_path = os.path.join(audio_dir, file_name)
        x, fs = librosa.load(file_path, sr=44100)
        return x, fs

    # change wave data to mel-stft
    def calculate_melsp(self, x, n_fft=1024, hop_length=128):
        stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
        log_stft = librosa.power_to_db(stft)
        melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
        return melsp

    # display wave in plots
    def show_wave(self, x):
        plt.plot(x)
        plt.show()

    # display wave in heatmap
    def show_melsp(self, melsp, fs):
        librosa.display.specshow(melsp, sr=fs)
        plt.colorbar()
        plt.show()

    # Augment audio data
    def add_white_noise(self, x, rate=0.002):
        return x + rate * np.random.randn(len(x))

    # data augmentation: shift sound in timeframe
    def shift_sound(self, x, rate=2):
        return np.roll(x, int(len(x)//rate))

    # data augmentation: stretch sound
    def stretch_sound(self, x, rate=1.1):
        input_length = len(x)
        x = librosa.effects.time_stretch(x, rate)
        if len(x)>input_length:
            return x[:input_length]
        else:
            return np.pad(x, (0, max(0, input_length - len(x))), "constant")


