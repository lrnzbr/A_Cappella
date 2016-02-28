import wave
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile # get the api
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd




handle = wave.open('audio/minuet.wav')
frame_rate = handle.getframerate()
n_frames = handle.getnframes()
window_size = 2* frame_rate
num_secs = int(math.ceil(n_frames/frame_rate))

snippet_list = []

print "Slicing Audio file..."
for i in xrange(num_secs):
    filename = 'audio/snippet' + str(i+1) + '.wav'
    snippet = wave.open(filename ,'wb')
    snippet.setnchannels(2)
    snippet.setsampwidth(handle.getsampwidth())
    snippet.setframerate(frame_rate)
    snippet.writeframes(handle.readframes(window_size))
    snippet_list.append(filename)
    handle.setpos(handle.tell() - 1 * frame_rate)
    snippet.close()

handle.close()


prints = []
print "Transforming to frequency domain...."
for i, item in enumerate(snippet_list):
    try:
        fs, data = wavfile.read(item) # load the data
        a = data.T[0] # this is a two channel soundtrack, I get the first track
        b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
        c = fft(b) # calculate fourier transform (complex numbers list)
        d = len(c)/2  # you only need half of the fft list (real signal symmetry)
        #plt.plot(abs(c[:(d-1)]),'r')
        #plt.show()
        thumbprint = abs(c[:(d-1)])

        prints.append(thumbprint)
    except ValueError:
        print "dead silence in track"
        pass


print "Fitting to model...."
y = np.linspace(0,len(prints)-1, len(prints))
y.reshape(1,-1)
X = pd.DataFrame(prints)
X = X.fillna(0)
rf = RandomForestClassifier()
rf.fit(X, y)

#predict
print "Predicting...."
sample = np.random.randint(0,len(prints))
sample_wav = 'audio/snippet' + str(sample+1) + '.wav'
fs, data = wavfile.read(sample_wav) # load the data
a = data.T[0] # this is a two channel soundtrack, I get the first track
b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
c = fft(b) # calculate fourier transform (complex numbers list)
d = len(c)/2  # you only need half of the fft list (real signal symmetry)
#plt.plot(abs(c[:(d-1)]),'r')
#plt.show()
sample_thumbprint = abs(c[:(d-1)])


sample_thumbprint = sample_thumbprint.reshape(1,-1)

guess = rf.predict(sample_thumbprint)

print "Given: ", sample
print "Predicted: ", guess
