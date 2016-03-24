from train import getModel
from sklearn.cross_validation import train_test_split, cross_val_score
import sys
import time
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint as sp_randint
# Utility functions to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return model.score(X_test, y_test), \
           precision_score(y_test, y_predict), \
           recall_score(y_test, y_predict)

'''Running tests for optimal classification algorithms'''

filename = 'accuracy_results.txt'
results = open('filename', 'a')
print "Starting Analysis...appending results to %s." % filename
results = open('accuracy_results.txt', 'a')
results.write("A Cappella Audio Classifier Scores\n")
results.write(time.strftime("%d/%m/%Y"))
results.write(time.strftime("%I:%M:%S"))



# results.write("\nTest 1:  Using FFT - Get Scores from all models\n")
#
# print "Unpacking X and y variables..."
# X = getModel('../models/X_fft.pkl')
# y = getModel('../models/y.pkl')
#
# X_train, X_test, y_train, y_test = train_test_split(X, y)
#
# try:
#     results.write("Model, Accuracy, Precision, Recall\n")
#     results.write("Random Forest:\n", get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=250, max_features=30))
#     results.write("\nLogistic Regression:\n", get_scores(LogisticRegression, X_train, X_test, y_train, y_test))
#     results.write("\nDecision Tree:\n", get_scores(DecisionTreeClassifier, X_train, X_test, y_train, y_test))
#     results.write("\nSVM:\n", get_scores(SVC, X_train, X_test, y_train, y_test))
# except:
#     print "TEST 1 FAILED!", sys.exc_info()[0]
#
# try:
#     results.write("\nTest 2: GridSearch on RandomForest with FFT\n")
#     clf = RandomForestClassifier()
#
#     # specify parameters and distributions to sample from
#     param_dist = {'n_estimators': sp_randint(10,100),
#                  "max_depth": [3, None],
#                  "max_features": sp_randint(1, 11),
#                  "min_samples_split": sp_randint(1, 11),
#                  "min_samples_leaf": sp_randint(1, 11),
#                  "bootstrap": [True, False],
#                  "criterion": ["gini", "entropy"]}
#
#     # run randomized search
#     n_iter_search = 40
#     random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                       n_iter=n_iter_search)
#
#     random_search.fit(X,y)
#
#     results.write("\nRandomizedSearchCV took %.2f seconds for %d candidates"
#           " parameter settings." % ((time() - start), n_iter_search))
#     report(random_search.grid_scores_)
#
#     # use a full grid over all parameters
#     param_grid = {"max_depth": [3, None],
#                   "max_features": [1, 3, 10],
#                   "min_samples_split": [1, 3, 10],
#                   "min_samples_leaf": [1, 3, 10],
#                   "bootstrap": [True, False],
#                   "criterion": ["gini", "entropy"]}
#
#     # run grid search
#     grid_search = GridSearchCV(clf, param_grid=param_grid)
#     start = time()
#     grid_search.fit(X, y)
#
#     results.write("\nGridSearchCV took %.2f seconds for %d candidate parameter settings.\n"
#           % (time() - start, len(grid_search.grid_scores_)))
#     results.write(report(grid_search.grid_scores_))
# except:
#     print "TEST 2 FAILED!", sys.exc_info()[0]


#results.write("\nTest 3: Using Peak Analysis - Get Scores from all models\n")

print "Unpacking X and y variables..."
X = getModel('../models/X_Peak_Analysis.pkl')
y = getModel('../models/y.pkl')

# X_train, X_test, y_train, y_test = train_test_split(X, y)
#
# print "Model, Accuracy, Precision, Recall\n"
# print "Random Forest:\n", get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=250, max_features=30)
# print "\nLogistic Regression:\n", get_scores(LogisticRegression, X_train, X_test, y_train, y_test)
# print "\nDecision Tree:\n", get_scores(DecisionTreeClassifier, X_train, X_test, y_train, y_test)
# print "\nSVM:\n", get_scores(SVC, X_train, X_test, y_train, y_test, )
# # except:
# #     print "TEST 3 Failed!" , sys.exc_info()
# #     pass

#try:

#results.write("Test 4: RandomForest GridSearch with Peak Analysis")
clf = RandomForestClassifier()

# specify parameters and distributions to sample from
param_dist = {'n_estimators': sp_randint(10,100),
             "max_depth": [3, None],
             "max_features": sp_randint(1, 11),
             "min_samples_split": sp_randint(1, 11),
             "min_samples_leaf": sp_randint(1, 11),
             "bootstrap": [True, False],
             "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 40
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                  n_iter=n_iter_search)

random_search.fit(X,y)

#print "\nRandomizedSearchCV took %.2f seconds for %d candidates" % ((time() - start), n_iter_search)
print" Parameter Settings \n"
report(random_search.grid_scores_)

# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

#results.write("\nGridSearchCV took %.2f seconds for %d candidate parameter settings.\n"
#      % (time() - start, len(grid_search.grid_scores_)))
#results.write(
report(grid_search.grid_scores_)
# except:
#     print "TEST 4 FAILED!", sys.exc_info()
#
# finally;
results.close
