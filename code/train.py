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

def chop_song(handle, folder):

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
        snippet_list.append([filename, i, convert_to_timestamp(i)])
        handle.setpos(handle.tell() - 1 * frame_rate)
        snippet.close()

    handle.close()
    del snippet_list[-1]
    return snippet_list

def transform_multiple(snippet_list):
    prints = []
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
            labels.append(item[1])
        except ValueError:
            print "dead silence in track"
            pass
    return prints, labels

    def transform_one_fft(item):
        fs, data = wavfile.read(item[0]) # load the data
        #print "data: ", data
        a = data.T[0] # this is a two channel soundtrack, It gets the first track
        b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
        c = fft(b) # calculate fourier transform (complex numbers list)
        d = len(c)/2  # you only need half of the fft list (real signal symmetry)
        #plt.plot(abs(c[:(d-1)]),'r')
        #plt.show()
        thumbprint = abs(c[:(d-1)])
        return thumbprint


# def transform_multiple_peak_analysis(snippet_list):
#     transformations = []
#     labels = []
#     for snippet in snippet_list:
#         transformation = single_file_featurization(snippet[0])
#         transformations.append(transformation)
#         labels.append(snippet[1])
#
#     return transformations, labels

'''FIX THIS LATER I guess'''
# def make_some_noise(prints, number_of_copies):
#     print "Adding Noise..."
#     for copy in xrange(number_of_copies):
#         for item in prints:
#             try:
#                 handle = wave.open(item[0])
#                 rate, data = scipy.io.wavfile.read(item[0])
#                 noise = np.random.normal(0,.01,len(data))
#                 data = pd.DataFrame(data, columns = ['left', 'right'])
#                 data['noise_left'] = (data['left'] + noise) / 2
#                 data['noise_right'] = (data['right'] + noise) / 2
#                 noise_sample = np.array(data['noise_left'], data['noise_right']).T
#                 filename = "noisy/noise"+str(np.random.randint(1,10000))+".wav"
#
#                 snippet = wave.open(filename ,'wb')
#                 snippet.setnchannels(2)
#                 snippet.setsampwidth(handle.getsampwidth())
#                 snippet.setframerate(frame_rate)
#                 snippet.writeframes(handle.readframes(window_size))
#
#
#                 scipy.io.wavfile.write(filename, rate, noise_sample)
#                 prints.append([filename, item[1], item[2]])
#             except ValueError:
#                 pass
#
#
#     return prints





def fit_rf(X, y):
    X = X.fillna(0)
    print "Fitting to random forest...."
    #dataframe = pd.DataFrame(prints)
    #y = dataframe[len(dataframe)-1]
    #y = y.reshape(1,-1)
    #X = dataframe[0:len(dataframe) - 2]
    #X = X.fillna(0)
    rf.fit(X, y)
    pickle_model(rf, 'rf')

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

if __name__ == '__main__':
    handle = wave.open('../audio/Adele_Solo.wav')

    snippets = chop_song(handle, "training")

    # handle2 = wave.open('Adele_Instrumental.wav')
    # snippets2 = chop_song(handle2, "training2")
    prints, labels = transform_multiple(snippets)
    #prints2, id2, labels2 = transform_multiple(snippets2)
    # fit_rf(prints)
    # print "Done!  Ready for Predictions"
    # Run all the other classifiers that we have learned so far in class
    X = pd.DataFrame(prints)
    #dataframe2 = pd.DataFrame(prints2)
    #dataframe = pd.concat([dataframe1,dataframe2])

    #labels1 = np.array(labels1)
    #labels2 = np.array(labels2)
    #labels = np.concatenate([labels1,labels2])

    y = labels
    #print "y looks like: ", y.head()


    #DO IT AGAIN WITH A DIFFERENT VERSION OF THE SONG:
    handle = wave.open('../audio/Adele_down.wav')
    snippets = chop_song(handle, "training2")
    prints, labels = transform_multiple(snippets)
    df = pd.DataFrame(prints)
    y = np.concatenate((y,labels), axis = 0)
    X = X.append(df)

    fit_rf(X,y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)
    #
    # def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
    #     model = classifier(**kwargs)
    #     model.fit(X_train, y_train)
    #     y_predict = model.predict(X_test)
    #     return model.score(X_test, y_test), \
    #            precision_score(y_test, y_predict), \
    #            recall_score(y_test, y_predict)
    #
    # print "    Model, Accuracy, Precision, Recall"
    # print "    Random Forest:", get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=25, max_features=5)
    # print "    Logistic Regression:", get_scores(LogisticRegression, X_train, X_test, y_train, y_test)
    # print "    Decision Tree:", get_scores(DecisionTreeClassifier, X_train, X_test, y_train, y_test)
    # print "    SVM:", get_scores(SVC, X_train, X_test, y_train, y_test)
    # print "    Naive Bayes:", get_scores(MultinomialNB, X_train, X_test, y_train, y_test)
    # ss = StandardScaler()
    # X = ss.fit_transform(X)
    #
    # lda = LinearDiscriminantAnalysis()
    #
    # X_lda = lda.fit_transform(X, y)


    # trains model using best performing model/hyperparameters using kfold grid search
    # svm = SVC(C=1, gamma=0.04)
    # svm.fit(X_lda, y)
    # svm.fit(X,y)
    # accuracy check to make sure the model is performing
    #y_pred_svm = svm.predict(X_lda)
    #y_pred_svm = svm.predict(X)
    #print 'model accuracy: ', accuracy_score(y, y_pred_svm)

    # cPickles models for later use
    # with open('../models/svm.pkl', 'wb') as f:
    #     cPickle.dump(svm, f)
    #
    # #with open('models/lda.pkl', 'wb') as f:
    #     #cPickle.dump(lda, f)
    #
    # with open('../models/ss.pkl', 'wb') as f:
    #     cPickle.dump(ss, f)

    print "Models ready!"
