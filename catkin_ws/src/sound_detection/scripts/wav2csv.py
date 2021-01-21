#!/usr/bin/env python
import sys
import os
import os
import pandas as pd
import glob
from scipy.io import wavfile

def wav2csv():
    input_filenames = sorted(glob.glob("wavfile/*.wav"))
    for input_filename in input_filenames:
        input_filename = input_filename.split("/")[-1]
        if input_filename[-3:] != 'wav':
            print("WARNING! Input file format should be *.wav")
            sys.exit()
        print(input_filename)

        samrate, data = wavfile.read("./wavfile/" + str(input_filename))
        print("Load is Done! \n")

        wavData = pd.DataFrame(data)
        if len(wavData.columns) == 2:
            wavData.columns = ['R', 'L']
            wavData.to_csv("csvfile/" + str(input_filename[:-4] + "_Output_stereo.csv"), mode="w")
            print("Save is done " + str(input_filename[:-4]) + "_Output_stereo.csv")

        elif len(wavData.columns) == 1:
            print("Mono .wav file \n")
            wavData.columns = ["M"]

            wavData.to_csv("csvfile/" + str(input_filename[:-4] + "_Output_mono.csv"), mode="w")
            print("Save is done " + str(input_filename[:-4]) + "_Output_mode.csv")

        else:
            print("Multi channel .wav file \n")
            print("Number of channel : " + len(wavData.columns) + "\n")
            wavData.to_csv("csvfile/" + str(input_file[:-4] + "Output_multi_channel.csv"), mode="w")
            print("Save is done " + str(input_filename[:-4]) + "_Output_multi_channel.csv")

if __name__ == "__main__":
    wav2csv()
