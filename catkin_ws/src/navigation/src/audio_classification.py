import os
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sn
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import model_selection
from sklearn import preprocessing
import IPython.display as ipd

base_dir = "./"
esc_dir = os.path.join(base_dir, "ESC-50-master")
meta_file = os.path.join(esc_dir, "meta/esc50.csv")
autio_dir = os.path.join(esc_dir, "audio/")
test_audio_dir = os.path.join(base_dir, "wavfile/")

try:
    meta_data = pd.read_csv(meta_file)
    import pprint
    pprint.pprint(meta_data)
except:
    pass

class Load_Dataset(object):
    def __init__(self):
        self.meta_data = pd.read_csv(meta_file)
        self.data_size = self.meta_data.shape

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

def preprocess(meta_data):
    x = list(meta_data.loc[:, "filename"])
    y = list(meta_data.loc[:, "target"])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, stratify=y)
    print("x train:{0}\ny train:{1}\nx test:{2}\ny test:{3}".format(len(x_train), len(y_train), len(x_test), len(y_test)))

    return x_train, x_test, y_train, y_test

freq = 128
time = 1723

def save_np_data(filename, x, y, aug=None, rates=None):
    np_data = np.zeros(freq * time * len(x)).reshape(len(x), freq, time)
    np_targets = np.zeros(len(y))
    for i in range(len(y)):
        _x, fs = load_wave_data(audio_dir, x[i])
        if aug is not None:
            _x = aug(x=_x, rate=rates[i])
        _x = calculate_melsp(_x)
        np_data[i] = _x
        np_targets[i] = y[i]
    np.savez(filename, x=np_data, y=np_targets)

if not os.path.exists("esc_melsp_test.npz"):
    save_np_data("esc_melsp_test.npz", x_test, y_test)

if not os.path.exists("esc_melsp_train_raw.npz"):
    save_np_data("esc_melsp_train_raw.npz", x_train, y_train)

if not os.path.exists("esc_melsp_train_wn.npz"):
    rates = np.random.randint(1, 50, len(x_train))/10000
    save_np_data("esc_melsp_train_wn.npz", x_train, y_train, aug=add_white_noise, rates=rates)

if not os.path.exists("esc_melsp_train_ss.npz"):
    rates = np.random.choice(np.arange(2,6),len(y_train))
    save_np_data("esc_melsp_train_ss.npz", x_train,  y_train, aug=shift_sound, rates=rates)

# save training dataset with stretch
if not os.path.exists("esc_melsp_train_st.npz"):
    rates = np.random.choice(np.arange(80,120),len(y_train))/100
    save_np_data("esc_melsp_train_st.npz", x_train,  y_train, aug=stretch_sound, rates=rates)

# save training dataset with combination of white noise and shift or stretch
if not os.path.exists("esc_melsp_train_com.npz"):
    np_data = np.zeros(freq*time*len(x_train)).reshape(len(x_train), freq, time)
    np_targets = np.zeros(len(y_train))
    for i in range(len(y_train)):
        x, fs = load_wave_data(audio_dir, x_train[i])
        x = add_white_noise(x=x, rate=np.random.randint(1,50)/1000)
        if np.random.choice((True,False)):
            x = shift_sound(x=x, rate=np.random.choice(np.arange(2,6)))
        else:
            x = stretch_sound(x=x, rate=np.random.choice(np.arange(80,120))/100)
        x = calculate_melsp(x)
        np_data[i] = x
        np_targets[i] = y_train[i]
    np.savez("esc_melsp_train_com.npz", x=np_data, y=np_targets)

classes = 50
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)

def cba(inputs, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

# define CNN
inputs = Input(shape=(x_train.shape[1:]))

x_1 = cba(inputs, filters=32, kernel_size=(1,8), strides=(1,2))
x_1 = cba(x_1, filters=32, kernel_size=(8,1), strides=(2,1))
x_1 = cba(x_1, filters=64, kernel_size=(1,8), strides=(1,2))
x_1 = cba(x_1, filters=64, kernel_size=(8,1), strides=(2,1))

x_2 = cba(inputs, filters=32, kernel_size=(1,16), strides=(1,2))
x_2 = cba(x_2, filters=32, kernel_size=(16,1), strides=(2,1))
x_2 = cba(x_2, filters=64, kernel_size=(1,16), strides=(1,2))
x_2 = cba(x_2, filters=64, kernel_size=(16,1), strides=(2,1))

x_3 = cba(inputs, filters=32, kernel_size=(1,32), strides=(1,2))
x_3 = cba(x_3, filters=32, kernel_size=(32,1), strides=(2,1))
x_3 = cba(x_3, filters=64, kernel_size=(1,32), strides=(1,2))
x_3 = cba(x_3, filters=64, kernel_size=(32,1), strides=(2,1))

x_4 = cba(inputs, filters=32, kernel_size=(1,64), strides=(1,2))
x_4 = cba(x_4, filters=32, kernel_size=(64,1), strides=(2,1))
x_4 = cba(x_4, filters=64, kernel_size=(1,64), strides=(1,2))
x_4 = cba(x_4, filters=64, kernel_size=(64,1), strides=(2,1))

x = Add()([x_1, x_2, x_3, x_4])

x = cba(x, filters=128, kernel_size=(1,16), strides=(1,2))
x = cba(x, filters=128, kernel_size=(16,1), strides=(2,1))

x = GlobalAveragePooling2D()(x)
x = Dense(classes)(x)
x = Activation("softmax")(x)

model = Model(inputs, x)

# initiate Adam optimizer
opt = keras.optimizers.adam(lr=0.00001, decay=1e-6, amsgrad=True)

# Let's train the model using Adam with amsgrad
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

if __name__ == "__main__":
    # audio_dir = "wavfile/"
    # file_name = "output1.wav" # for test
    audio_classifier = Load_Dataset()
    x, fs = audio_classifier.load_wave_data(audio_dir, file_name)
    melsp = audio_classifier.calculate_melsp(x)
    # audio_classifier.show_wave(x)
    # audio_classifier.show_melsp(melsp, fs)

    x_1 = audio_classifier.add_white_noise(x)
    x_2 = audio_classifier.shift_sound(x)
    x_3 = audio_classifier.stretch_sound(x)
    # audio_classifier.show_wave(x_3)
