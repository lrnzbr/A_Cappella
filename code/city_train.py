from setup import LOCAL_REPO_DIR
import numpy as np
import pandas as pd
import cPickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report

def save_pickle(model):
    '''
    INPUT:
    path of pickled model
    OUTPUT:
    unpickled model
    Pass in the location of the pickled model and the function will unpickle it and return
    the opened model
    '''

    with open(LOCAL_REPO_DIR + 'models/' + str(model) + '.pkl', 'wb') as f:
        return cPickle.dump(model, f)

def train_model(csv_path):
    '''
    INPUT:
    audio features csv with 'class' labels included
    OUTPUT:
    three pickled models stored in the models dir
    - StandardScaler (sklearn)
    - LinearDiscriminantAnalysis (sklearn)
    - SVC (sklearn)
    Takes an audio feature csv (created from 'feature_extraction.py') and returns pickled models to use
    '''
    csv = LOCAL_REPO_DIR + csv_path
    df = pd.read_csv(csv_path)

    # extracts X, y for training model from dataframe
    X = df.drop(['class', 'fold', 'Unnamed: 0'], axis=1).values
    y = df['class'].values

    # feature matrix has many different scales, need to standardize
    ss = StandardScaler()
    X = ss.fit_transform(X)

    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X, y)

    # trains model using best performing model/hyperparameters using kfold grid search
    svm = SVC(C=1, gamma=0.04)
    svm.fit(X_lda, y)

    # accuracy check to make sure the model is performing
    y_pred_svm = svm.predict(X_lda)
    print 'model accuracy: ', accuracy_score(y, y_pred_svm)

    # cPickles models for later use
    with open(LOCAL_REPO_DIR + 'model/svm.pkl', 'wb') as f:
        cPickle.dump(svm, f)

    with open(LOCAL_REPO_DIR + 'model/lda.pkl', 'wb') as f:
        cPickle.dump(lda, f)

    with open(LOCAL_REPO_DIR + 'model/ss.pkl', 'wb') as f:
        cPickle.dump(ss, f)

if __name__ == '__main__':
    train_model('csv/citysounds.csv')
