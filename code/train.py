import wave
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile # get the api
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import cPickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import train_test_split, cross_val_score
import scipy

import numpy as np
import pandas as pd
import librosa as lr
from scipy.stats import skew, kurtosis
import scipy.io.wavfile as wav

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from shutil import copyfile
import os


from scipy.io import wavfile
from scipy.io.wavfile import write

import wave
from allantools import noise


import shutil

'''
STEPS FOR TRAINING:

1.  Copy the input audio file to the 'transformed' folder

2.  Add white noise and pink noise to the original file

3.  Add pitch bend for one octave up and down to all files.

4. Chop all files into snippets.

5.  Apply Transformation to each snippet

6.(optional) Make a Test Train split of the data

7. Fit to the model of choice

8.(optional) check accuracy of model

9. Pickle Model

10. Cleanup - delete snippets and transformations

'''

# variables for the mfcc extraction

# fft window size: ex. sample rate is 44.1 kHz, a window of 1024 is 0.23 ms
window = 1024
# hop size: how far to jump ahead to the next window
hop = window / 2
# number of mel frequency triangular filters
n_filters = 40
# number of mfcc coefficients to return
n_coeffs = 25
# sample rate of the audio before transformation into the frequency domain
sample_rate = 44100

rf = RandomForestClassifier()

def chop_song(filename, folder):
    handle = wave.open(filename, 'rb')
    frame_rate = handle.getframerate()
    n_frames = handle.getnframes()
    window_size = 2 * frame_rate
    num_secs = int(math.ceil(n_frames/frame_rate))

    snippet_list = []

    print "Slicing Audio file..."
    for i in xrange(num_secs):
        filename = '../audio/' + folder + '/snippet'+ str(i+1) + '.wav'
        snippet = wave.open(filename ,'wb')
        snippet.setnchannels(2)
        snippet.setsampwidth(handle.getsampwidth())
        snippet.setframerate(frame_rate)
        snippet.writeframes(handle.readframes(window_size))
        snippet_list.append([filename, convert_to_timestamp(i)])
        handle.setpos(handle.tell() - 1 * frame_rate)
        snippet.close()

    handle.close()
    del snippet_list[-1]
    return snippet_list

def transform_multiple_fft(snippet_list):
    transformations = []
    #ids = []
    labels = []
    transform = pd.DataFrame()
    print "Transforming to frequency domain...."
    for i, item in enumerate(snippet_list):
        try:
            fs, data = wavfile.read(item[0]) # load the data
            #print "data: ", data
            a = data.T[0] # this is a two channel soundtrack, It gets the first track
            b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
            c = fft(b) # calculate fourier transform (complex numbers list)
            d = len(c)/2  # you only need half of the fft list (real signal symmetry)
            #plt.plot(abs(c[:(d-1)]),'r')
            #plt.show()
            thumbprint = abs(c[:(d-1)])
            prints.append(thumbprint)
            #ids.append(item[1])
            labels.append(item[1])
        except ValueError:
            print "dead silence in track"
            pass
    return transformations,labels, #ids #

def transform_multiple_peak_analysis(snippet_list):
    transformations = []
    labels = []
    for snippet in snippet_list:
        transformation = single_file_featurization(snippet[0])
        transformations.append(transformation)
        labels.append(snippet[1])

    return transformations, labels



def fit_rf(X, y, model_name='rf'):
    print "Fitting to random forest...."
    # dataframe = pd.DataFrame(prints)
    # y = dataframe[len(dataframe)-1]
    # y = y.reshape(1,-1)
    # X = dataframe[0:len(dataframe) - 2]
    # X = X.fillna(0)
    rf.fit(X, y)
    pickle_model(rf, model_name)

def fit_svm(prints):
    print "Fitting to SVM...."
    dataframe = pd.DataFrame(prints)
    y = dataframe[2]
    X = dataframe[0]

    # in case feature matrix has many different scales, need to standardize
    ss = StandardScaler()
    X = ss.fit_transform(X)


    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X_1, y)

    # trains model using best performing model/hyperparameters using kfold grid search
    svm = SVC(C=1, gamma=0.04)
    svm.fit(X_lda, y)

    pickle_model(svm, 'svm')


def getModel(pickle_path):
    with open(pickle_path, 'rb') as f:
        return cPickle.load(f)


def pickle_model(model, modelname):
    with open('../models/' + str(modelname) + '.pkl', 'wb') as f:
        return cPickle.dump(model, f)

def convert_to_timestamp(y_val):
    minutes = y_val / 60
    seconds = y_val - (minutes * 60)
    if minutes < 10:
        if seconds < 10:
            return "0"+str(minutes)+":"+"0"+ str(seconds)
        else:
            return "0"+str(minutes)+":"+str(seconds)
    else:
        if seconds < 10:
            return str(minutes)+":"+"0"+str(seconds)
        else:
            return str(minutes) + ":" + str(seconds)
def single_file_featurization(wavfile):
    '''
    INPUT:
    row of dataframe with 'audio_slice_name' as the filename of the audio sample

    OUTPUT:
    feature vector for audio sample

    Function for dataframe apply for extracting each audio sample into a feature vector
    of mfcc coefficients
    '''

    # print statements to update the progress of the processing
    print wavfile
    try:
        # load the raw audio .wav file as a matrix using librosa
        wav_mat, sr = lr.load(wavfile, sr=sample_rate)

        # create the spectrogram using the predefined variables for mfcc extraction
        S = lr.feature.melspectrogram(wav_mat, sr=sr, n_mels=n_filters, fmax=sr/2, n_fft=window, hop_length=hop)

        # using the pre-defined spectrogram, extract the mfcc coefficients
        mfcc = lr.feature.mfcc(S=lr.logamplitude(S), n_mfcc=25)

        # calculate the first and second derivatives of the mfcc coefficients to detect changes and patterns
        mfcc_delta = lr.feature.delta(mfcc)
        mfcc_delta = mfcc_delta.T
        mfcc_delta2 = lr.feature.delta(mfcc, order=2)
        mfcc_delta2 = mfcc_delta2.T
        mfcc = mfcc.T

        # combine the mfcc coefficients and their derivatives in a column stack for analysis
        total_mfcc = np.column_stack((mfcc, mfcc_delta, mfcc_delta2))

        # use the average of each column to condense into a feature vector
        # this makes each sample uniform regardless of the length of original the audio sample
        # the following features are extracted
        # - avg of mfcc, first derivative, second derivative
        # - var of mfcc, first derivative, second derivative
        # - max of mfcc
        # - min of mfcc
        # - median of mfcc
        # - skew of mfcc
        # - kurtosis of mfcc
        avg_mfcc = np.mean(total_mfcc, axis=0)
        var_mfcc = np.var(total_mfcc, axis=0)
        max_mfcc = np.max(mfcc, axis=0)
        min_mfcc = np.min(mfcc, axis=0)
        med_mfcc = np.median(mfcc, axis=0)
        skew_mfcc = skew(mfcc, axis=0)
        kurt_mfcc = skew(mfcc, axis=0)

        # combine into one vector and append to the total feature matrix
        return np.concatenate((avg_mfcc, var_mfcc, max_mfcc, min_mfcc, med_mfcc, skew_mfcc, kurt_mfcc))
    except:
        print "Uhmmm something bad happened"
        return np.zeros(7)
def speedx(sound_array, factor):
    """ Multiplies the sound's speed by some `factor` """
    indices = np.round( np.arange(0, len(sound_array), factor) )
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[ indices.astype(int) ]

def stretch(sound_array, f, window_size, h):
    """ Stretches the sound by a factor `f` """

    phase  = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros( len(sound_array) /f + window_size)

    for i in np.arange(0, len(sound_array)-(window_size+h), h*f):

        # two potentially overlapping subarrays
        a1 = sound_array[i: i + window_size]
        a2 = sound_array[i + h: i + window_size + h]

        # resynchronize the second array on the first
        s1 =  np.fft.fft(hanning_window * a1)
        s2 =  np.fft.fft(hanning_window * a2)
        phase = (phase + np.angle(s2/s1)) % 2*np.pi
        a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))

        # add to result
        i2 = int(i/f)
        result[i2 : i2 + window_size] += hanning_window*a2_rephased.astype(result[i2 : i2 + window_size].dtype)

    result = ((2**(16-4)) * result/result.max()) # normalize (16bit)

    return result.astype('int16')

def pitchshift(filename, n, window_size=2**13, h=2**11):
    """ Changes the pitch of a sound by ``n`` semitones. """
    if n == 0:
        return
    print "Shifting  by %d steps" % n

    fps, snd_array = wavfile.read(filename)
    factor = 2**(1.0 * n / 12.0)
    snd_array1 = snd_array[:,0]
    snd_array2 = snd_array[:,1]

    stretched1 = stretch(snd_array1, 1.0/factor, window_size, h)
    stretched2 = stretch(snd_array2, 1.0/factor, window_size, h)
    stretched = np.array([stretched1, stretched2]).T
    filename, file_extension = os.path.splitext(filename)
    filename = filename+"pitchshift"+str(n)+".wav"

    write(filename, 44100, speedx(stretched[window_size:], factor))

    return




def create_noisy_set(filename):

    print "Creating noisy copies...\n"

    add_white_noise(filename)
    add_pink_noise(filename)
    add_violet_noise(filename)
    add_brown_noise(filename)

def add_white_noise(filename):
    print "Adding White Noise..."
    fps, snd_array = wavfile.read(filename)
    noise1 = noise.white(len(snd_array))
    noise1 = map(lambda x: int(x/.01), noise1)

    noise2 = np.array([noise1,noise1]).T
    noisy_array = noise2 + snd_array
    filename, file_extension = os.path.splitext(filename)
    filename = "../audio/transformed/"+filename+"whitenoise.wav"

    scaled = np.int16(noisy_array/np.max(np.abs(noisy_array)) * 32767)
    write(filename, 44100, scaled)
    return

def add_pink_noise(filename):
    print "Adding Pink Noise..."
    fps, snd_array = wavfile.read(filename)
    noise1 = noise.pink(len(snd_array))
    noise1 = map(lambda x: int(x/.01), noise1)

    noise2 = np.array([noise1,noise1]).T
    noisy_array = noise2 + snd_array
    filename, file_extension = os.path.splitext(filename)
    filename = "../audio/transformed/"+filename+"pinknoise.wav"

    scaled = np.int16(noisy_array/np.max(np.abs(noisy_array)) * 32767)
    write(filename, 44100, scaled)
    return


def add_violet_noise(filename):
    print "Adding Violet Noise..."
    fps, snd_array = wavfile.read(filename)
    noise1 = noise.white(len(snd_array))
    noise1 = map(lambda x: int(x/.01), noise1)

    noise2 = np.array([noise1,noise1]).T
    noisy_array = noise2 + snd_array
    filename, file_extension = os.path.splitext(filename)
    filename = "../audio/transformed/"+filename+"violetnoise.wav"

    scaled = np.int16(noisy_array/np.max(np.abs(noisy_array)) * 32767)
    write(filename, 44100, scaled)
    return

def add_brown_noise(filename):
    def add_noise_white(filename):
        print "Adding Brown Noise..."
        fps, snd_array = wavfile.read(filename)
        noise1 = noise.brown(len(snd_array))
        noise1 = map(lambda x: int(x/.01), noise1)

        noise2 = np.array([noise1,noise1]).T
        noisy_array = noise2 + snd_array
        filename, file_extension = os.path.splitext(filename)
        filename = "../audio/transformed/"+filename+"brownnoise.wav"

        scaled = np.int16(noisy_array/np.max(np.abs(noisy_array)) * 32767)
        write(filename, 44100, scaled)
        return




if __name__ == '__main__':
    print "Starting training...\n"
    filename = '../audio/originals/Adele_Solo.wav'
    '''Make chopped and transformed directories'''
    os.mkdir('../audio/transformed')
    os.mkdir('../audio/chopped')
    '''Copy file into the 'transformations' folder'''
    copyfile(filename, '../audio/transformed/original.wav')

    rootdir = '../audio/transformed'

    '''pitch bend 8 steps up and down from the origninal'''
    for i in range(-8,8):
        pitchshift('../audio/transformed/origninal.wav', i)

    '''add noisy copies'''
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".wav"):
            create_noisy_set(filename)

    print "TEST 1: FFT:"
    '''Chop and Transform each track'''
    X = pd.DataFrame()
    y = []
    for filename in os.listdir('../audio/transformed'):
        if filename.endswith(".wav"):
            snippets = (chop_song('../audio/transformed/'+ filename, "chopped"))
            prints, labels = transform_multiple_fft(snippets)
            X = X.append(pd.DataFrame(prints))
            y = np.concatenate((y,labels), axis = 0)

    #SAVE THE MODEL:
    fit_rf(X,y,'rf_fft')
    '''Cross Validate'''
    X_train, X_test, y_train, y_test = train_test_split(X, y)


    def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
        model = classifier(**kwargs)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        return model.score(X_test, y_test), \
               precision_score(y_test, y_predict), \
               recall_score(y_test, y_predict)

    print "    Model, Accuracy, Precision, Recall"
    print "    Random Forest:", get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=25, max_features=5)
    print "    Logistic Regression:", get_scores(LogisticRegression, X_train, X_test, y_train, y_test)
    print "    Decision Tree:", get_scores(DecisionTreeClassifier, X_train, X_test, y_train, y_test)
    print "    SVM:", get_scores(SVC, X_train, X_test, y_train, y_test)
    #print "    Naive Bayes:", get_scores(MultinomialNB, X_train, X_test, y_train, y_test)


    print "Test 2: Using Peak Analysis"
    '''Chop and Transform each track'''
    X = pd.DataFrame()
    y = []
    for filename in os.listdir('../audio/transformed'):
        if filename.endswith(".wav"):
            snippets = (chop_song('../audio/transformed/'+ filename, "chopped"))
            prints, labels = transform_multiple_peak_analysis(snippets)
            X = X.append(pd.DataFrame(prints))
            y = np.concatenate((y,labels), axis = 0)

    #SAVE THE MODEL:
    fit_rf(X,y,'rf_peak_analysis')
    '''Cross Validate'''
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print "    Model, Accuracy, Precision, Recall"
    print "    Random Forest:", get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=25, max_features=5)
    print "    Logistic Regression:", get_scores(LogisticRegression, X_train, X_test, y_train, y_test)
    print "    Decision Tree:", get_scores(DecisionTreeClassifier, X_train, X_test, y_train, y_test)
    print "    SVM:", get_scores(SVC, X_train, X_test, y_train, y_test)




    '''

    #
    # ss = StandardScaler()
    # X = ss.fit_transform(X)
    #
    # lda = LinearDiscriminantAnalysis()
    #
    # X_lda = lda.fit_transform(X, y)
    #
    #
    # # trains model using best performing model/hyperparameters using kfold grid search
    # svm = SVC(C=1, gamma=0.04)
    # #svm.fit(X_lda, y)
    # svm.fit(X,y)
    # # accuracy check to make sure the model is performing
    # #y_pred_svm = svm.predict(X_lda)
    # y_pred_svm = svm.predict(X)
    # print 'model accuracy: ', accuracy_score(y, y_pred_svm)
    #
    #
    #
    # # cPickles models for later use
    # with open('../models/svm.pkl', 'wb') as f:
    #     cPickle.dump(svm, f)
    #
    # #with open('models/lda.pkl', 'wb') as f:
    #     #cPickle.dump(lda, f)
    #
    # with open('../models/ss.pkl', 'wb') as f:
    #     cPickle.dump(ss, f)
    '''
    print "Models ready!....Cleaning up"
    shutil.rmtree('../audio/transformed/')
    shutil.rmtree('../audio/chopped/')

    print "Done!"
