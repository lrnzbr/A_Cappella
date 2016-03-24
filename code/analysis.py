import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pdb

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')
from train import getModel, pickle_model



def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return model.score(X_test, y_test), \
           precision_score(y_test, y_predict), \
           recall_score(y_test, y_predict)


'''Running tests for optimal classification algorithms'''


'''LOAD THE DATA'''
print "Unpacking X and y variables..."
X = getModel('../models/X_Peak_Analysis.pkl')
#X = preprocessing.scale(X)
y = getModel('../models/y.pkl')
y = y[0:len(X)]
print len(X), len(y)
'''SPLIT DATA FOR CROSS VALIDATION'''

X_train, X_test, y_train, y_test = train_test_split(X, y)



'''******Support Vector Machines***'''
print "Linear SVM: ACCURACY, PRECISION, RECALL"

print get_scores(SVC, X_test, X_test, y_train, y_test, kernel='linear')

# print "RBF SVM:"
# print get_scores(SVC, X_train, X_test, y_train, y_test, kernel='rbf')
#
# print "Signmoid SVM:"
# print get_scores(SVC, X_train, X_test, y_train, y_test, kernel='sigmoid')


'''********LOGISTIC REGRESSION*******'''
print "Logistic Regression: "

print get_scores(LogisticRegression, X_train, X_test, y_train, y_test)
#
# '''********RANDOM FORESTS********'''
# print "Random Forest"
# print get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=100)

# print "Extra Trees"
# print get_scores(ExtraTreesClassifier, X_train, X_test, y_train, y_test)
#
# print "XGBoost"
# print get_scores(XGBClassifier, X_train, X_test, y_train, y_test)

# rf = RandomForestClassifier(n_estimators=100, n_jobs = -1)
# rf.fit(X, y)
# pickle_model(rf, "RF_2Sec_PeakAn")
#
# lr = LogisticRegression()
# lr.fit(X,y)
# pickle_model(lr, "LR_2sec_LogReg")
#
# svm = SVC(kernel='linear')
# svm.fit(X,y)
# pickle_model(svm, 'svm_2sec_peakAn')
