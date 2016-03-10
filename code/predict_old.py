import wave
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile # get the api
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave
import copy

from train import chop_song, getModel, transform_multiple

THRESHOLD = 300  # audio levels not normalised.
CHUNK_SIZE = 1024
SILENT_CHUNKS = 3 * 44100 / 1024  # about 3sec
FORMAT = pyaudio.paInt16
FRAME_MAX_VALUE = 2 ** 15 - 1
NORMALIZE_MINUS_ONE_dB = 10 ** (-1.0 / 20)
RATE = 44100
CHANNELS = 2
TRIM_APPEND = RATE / 4


def is_silent(data_chunk):
    "Returns 'True' if below the 'silent' threshold"
    return max(data_chunk) < THRESHOLD
def normalize(data_all):
    """Amplify the volume out to max -1dB"""
    # MAXIMUM = 16384
    normalize_factor = (float(NORMALIZE_MINUS_ONE_dB * FRAME_MAX_VALUE)
                        / max(abs(i) for i in data_all))

    r = array('h')
    for i in data_all:
        r.append(int(i * normalize_factor))
    return r

def trim(data_all):
    _from = 0
    _to = len(data_all) - 1
    for i, b in enumerate(data_all):
        if abs(b) > THRESHOLD:
            _from = max(0, i - TRIM_APPEND)
            break

    for i, b in enumerate(reversed(data_all)):
        if abs(b) > THRESHOLD:
            _to = min(len(data_all) - 1, len(data_all) - 1 - i + TRIM_APPEND)
            break

    return copy.deepcopy(data_all[_from:(_to + 1)])


def record():
    """Record a word or words from the microphone and
    return the data as an array of signed shorts."""

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,frames_per_buffer=CHUNK_SIZE)

    silent_chunks = 0
    audio_started = False
    data_all = array('h')
    i = 0
    print "recording..."

    while i < 400:
        # little endian, signed short

        data_chunk = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            data_chunk.byteswap()
        data_all.extend(data_chunk)

        silent = is_silent(data_chunk)

        if audio_started:
            if silent:
                print "silent"
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            else:
                silent_chunks = 0
        elif not silent:
            print "not silent"
            audio_started = True
        i += 1

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    data_all = trim(data_all)  # we trim before normalize as threshhold applies to un-normalized wave (as well as is_silent() function)
    data_all = normalize(data_all)
    return sample_width, data_all

if __name__ == '__main__':
    print "Waiting for audio input...."
    #Take in microphone input and chop into 2 second intervals
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)
    path = 'test.wav'
    wf = wave.open(path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
    wave_read = wave.open(path)
    song_pieces = chop_song(wave_read,"sample")
    fingerprints = transform_multiple(song_pieces)
    X_test = pd.Series(fingerprints).fillna(0)
    predictions = []
    model = getModel('models/rf.pkl')
    for fp in X_test:
        if len(fp) >= len(X_test[0]):
            predictions.append(model.predict(fp))
    print "time predictions:", predictions





# #predict
# print "Predicting...."
# sample = np.random.randint(0,len(prints))
# sample_wav = 'audio/snippet' + str(sample+1) + '.wav'
# #sample_wav = 'test_output.wav'
# fs, data = wavfile.read(sample_wav) # load the data
# a = data.T[0] # this is a two channel soundtrack, I get the first track
# b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
# c = fft(b) # calculate fourier transform (complex numbers list)
# d = len(c)/2  # you only need half of the fft list (real signal symmetry)
# #plt.plot(abs(c[:(d-1)]),'r')
# #plt.show()
# sample_thumbprint = abs(c[:(d-1)])
#
#
# sample_thumbprint = sample_thumbprint.reshape(1,-1)
#
# guess = rf.predict(sample_thumbprint)
#
# print "Given:", sample
# print "Predicted: ", guess
