#!/usr/bin/env python
import pyaudio
import wave
import rospy

def Record():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000 # can kinect deal with 44100hz?
    RECORD_SECONDS = 20
    SAMPLE_SECONDS = 5
    INDEX = 4

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=INDEX,
                frames_per_buffer=CHUNK)

    print("* recording")

    for i in range(0, int(RECORD_SECONDS / SAMPLE_SECONDS)):
        frames = []
        WAVE_OUTPUT_FILENAME = "output" + str(i) + ".wav"
        for j in range(0, int(RATE / CHUNK * SAMPLE_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("* done recording")
        if i == (RECORD_SECONDS / SAMPLE_SECONDS):
            stream.stop_stream()
            stream.close()
            p.terminate()

        wf = wave.open("./wavfile/" + WAVE_OUTPUT_FILENAME, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

if __name__ == "__main__":
    rospy.init_node("record")
    Record()
