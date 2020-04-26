# You may want to install "gprof2dot"
import io
from collections import Counter

import numpy as np
import pandas as pd
import scipy.io
import sklearn.model_selection
import sklearn.tree
from numpy import genfromtxt
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle
import math
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier # I promise its only for Kaggle

import pydot

eps = 1e-5  # a small number
np.random.seed(123456)


class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None, m=0, stopping_criterion=0):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes
        self.m = m # if m is nonzero, we will select features to split on from a random subset of m features
        self.stopping=stopping_criterion

    @staticmethod
    def information_gain(X, y, thresh):
        """Returns the information gain of making a given split"""
        H_bef = DecisionTree.gini_impurity(y) #gini impurity before split
        yl = []
        yr = []
        for i in range(len(X)):
            val = X[i]
            if val < thresh:
                yl.append(val)
            else:
                yr.append(val)
        H_aft = (len(yl)*DecisionTree.gini_impurity(yl) + len(yr)*DecisionTree.gini_impurity(yr)) / len(y) #gini impurity after split
        return H_bef - H_aft

    @staticmethod
    def gini_impurity(y):
        """returns the probability of drawing a random label and classifying it incorrectly
           using the class distribution"""
        if len(y) == 0:
            return 0
        num1 = sum(y) #number of ones
        num0 = len(y)-sum(y) #number of zeros
        return (2*num0*num1)/(len(y)*len(y))

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def getFeatureIndices(self, num_features):
        """Returns a range/set of indices corresponding to the randomly selected subset of features chosen for a split
        num-features is the number of features we have to select from, self.m is how many features we want (unless m=0)"""
        if self.m: # random forest
            feature_indices = set()
            while len(feature_indices) < self.m:
                feature_indices.add(np.random.randint(num_features))
            return feature_indices
        else: # no bagging or random forests
            return range(num_features)


    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            feature_indices = self.getFeatureIndices(X.shape[1]) #returns bagged data & labels and a set of the indices for features we are considering at this split
            thresh = np.array([
                np.linspace(np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                if not i in feature_indices: # i is not a feature we are considering
                    gains.append([-1*float('inf') for t in thresh[i, :]])
                else:
                    gains.append([self.information_gain(X[:, i], y, t) for t in thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0 and len(y) > self.stopping:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features, m=self.m)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features, m=self.m)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    
    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat


"""class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            sklearn.tree.DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO implement function
        for tree in self.decision_trees:
            tree.fit()

    def predict(self, X):
        # TODO implement function
        pass"""


"""class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        # TODO implement function
        pass"""

def getBaggedData(X, y):
    """returns bagged data for inputs X and y"""
    size = len(X)
    xPrime = []
    yPrime = []
    for _ in range(size):
        rand = np.random.randint(size)
        xPrime.append(X[rand])
        yPrime.append(y[rand])
    return np.array(xPrime), np.array(yPrime)

class RandomForest:
    def __init__(self, max_depth=3, feature_labels=None, m=0, stopping_criterion=0, num_trees=200):
        self.num_trees = num_trees
        self.decision_trees = [DecisionTree(max_depth=max_depth, feature_labels=feature_labels, m=m, stopping_criterion=stopping_criterion) for _ in range(num_trees)]

    def fit(self, X, y):
        for t in self.decision_trees:
            xPrime, yPrime = getBaggedData(X,y)
            t.fit(xPrime, yPrime)

    def predict(self, X):
        individual_predictions = [t.predict(X) for t in self.decision_trees]
        sum_predictions = sum(individual_predictions)
        predictions = []
        for pred in sum_predictions:
            if pred > self.num_trees/2:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions


class BoostedRandomForest(RandomForest):
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        # TODO implement function
        return self

    def predict(self, X):
        # TODO implement function
        pass


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(np.float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=np.float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode[0]
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", sklearn.model_selection.cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        print("First splits", first_splits)

def results_to_csv(y_test, name):
    y_test = np.array(y_test)
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1. 
    df.to_csv(name, index_label='Id')

def accuracy(preds, vals):
    """Returns accuracy of predictions"""
    if len(preds) == 0:
        return 0
    num_correct = 0
    for i in range(len(preds)):
        if preds[i] == vals[i]:
            num_correct += 1
    return num_correct/len(preds)

# Helper code to initialize the data
def splitSpamData():
    """returns various splits of the data (purpose is to avoid repetitive code
    returns in the following order: (X, y, train_data, train_labels, val_data, val_labels, test_data, features)"""
    features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription", "creative",
            "height", "featured", "differ", "width", "other", "energy", "business", "message",
            "volumes", "revision", "path", "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis", "square_bracket",
            "ampersand"
        ]
    assert len(features) == 32
    path_train = 'spam_data.mat'
    data = scipy.io.loadmat(path_train)
    shuffled_data, shuffled_labels = shuffle(data["training_data"], np.squeeze(data["training_labels"]))
    split_val = int(len(shuffled_data) * 0.8)
    training_data = shuffled_data[:split_val]
    training_values = shuffled_labels[:split_val]
    val_data = shuffled_data[split_val:]
    val_labels = shuffled_labels[split_val:]
    test_data = data["test_data"]
    return shuffled_data, shuffled_labels, training_data, training_values, val_data, val_labels, test_data, features

def splitTitanicData():
    """returns various splits of the data (purpose is to avoid repetitive code
    returns in the following order: (X, y, train_data, train_labels, val_data, val_labels, test_data, features)"""
    path_train = 'titanic_training.csv'
    data = genfromtxt(path_train, delimiter=',', dtype=None)
    path_test = 'titanic_testing_data.csv'
    test_data = genfromtxt(path_test, delimiter=',', dtype=None)
    y = data[1:, 0]  # label = survived
    class_names = ["Died", "Survived"]

    labeled_idx = np.where(y != b'')[0]
    y = np.array(y[labeled_idx], dtype=np.int)
    print("\n\nPart (b): preprocessing the titanic dataset")
    X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
    X = X[labeled_idx, :]
    Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
    assert X.shape[1] == Z.shape[1]
    features = list(data[0, 1:]) + onehot_features
    data, labels = shuffle(X, y)
    split_val = int(len(data) *.8)
    training_data = data[:split_val]
    training_labels = labels[:split_val]
    val_data = data[split_val:]
    val_labels = labels[split_val:]
    return X, y, training_data, training_labels, val_data, val_labels, Z, features
    
# Code for 3.1

def useDecisionTree(dataset):
    """Uses a Decision Tree to predict the test labels, returns the predicted test labels"""
    print("Beginning evaluation of DecisionTree implementation...")
    # Set up the data
    if dataset == 'spam':
        X, y, training_data, training_labels, val_data, val_labels, test_data, features = splitSpamData()
        max_depth = 18
        stopping = 10
    else:
        X, y, training_data, training_labels, val_data, val_labels, test_data, features = splitTitanicData()
        max_depth = 13
        stopping = 2
    # Let's select the best max_depth
    """best_acc = 0
    for i in range(1, len(features)):
        dt = DecisionTree(max_depth=i, feature_labels=features)
        dt.fit(training_data, training_labels)
        preds = dt.predict(val_data)
        acc = accuracy(preds, val_labels)
        print("Accuracy of max_depth " + str(i) + " = " + str(acc))
        if acc > best_acc:
            best_acc = acc
            max_depth = i"""
    dt = DecisionTree(max_depth=max_depth, feature_labels=features, stopping_criterion=stopping)
    dt.fit(training_data, training_labels)
    train_preds = dt.predict(training_data)
    train_acc = accuracy(train_preds, training_labels)
    val_preds = dt.predict(val_data)
    val_acc = accuracy(val_preds, val_labels)
    print("Decision Tree on the " + dataset + " dataset.")
    print("Training accuracy achieved with max depth = " + str(max_depth) + ": " + str(train_acc))
    print("Validation accuracy achieved with max depth = " + str(max_depth) + ": " + str(val_acc))
    dt = DecisionTree(max_depth=max_depth, feature_labels=features) #reinitialize
    dt.fit(X, y) #train on all data
    test_preds = dt.predict(test_data)
    return test_preds

# Code for 3.2

def useRandomForest(dataset):
    """Uses a Random Forest to predict the test labels, returns the predicted test labels"""
    print("Beginning evaluation of RandomForest implementation...")
    # Set up the data
    if dataset == 'spam':
        X, y, training_data, training_labels, val_data, val_labels, test_data, features = splitSpamData()
        max_depth = 18
        stopping = 10
    else:
        X, y, training_data, training_labels, val_data, val_labels, test_data, features = splitTitanicData()
        max_depth = 12
        stopping = 2
    # Let's select the best max_depth
    num_trees = 200
    rf = RandomForest(max_depth=max_depth, feature_labels=features, m=6, stopping_criterion=stopping, num_trees=num_trees)
    rf.fit(training_data, training_labels)
    train_preds = rf.predict(training_data)
    train_acc = accuracy(training_labels, train_preds)
    val_preds = rf.predict(val_data)
    val_acc = accuracy(val_preds, val_labels)
    print("Random Forest on the " + dataset + " dataset.")
    print("Training accuracy achieved with max depth = " + str(max_depth) + " and " + str(num_trees) + " random trees: " + str(train_acc))
    print("Validation accuracy achieved with max depth = " + str(max_depth) + " and " + str(num_trees) + " random trees: " + str(val_acc))
    rf = RandomForest(max_depth=max_depth, feature_labels=features, m=6, num_trees=num_trees) #reinitialize
    rf.fit(X, y) #train on all data
    test_preds = rf.predict(test_data)
    return test_preds


def plotValAccuracies():
    # Set up the data
    X, y, training_data, training_labels, val_data, val_labels, test_data, features = splitSpamData()
    # Let's select the best max_depth
    x_ax = [i for i in range(1,41)]
    y_ax = []
    for i in x_ax:
        print(i)
        dt = DecisionTree(max_depth=i)
        dt.fit(training_data, training_labels)
        preds = dt.predict(val_data)
        acc = accuracy(preds, val_labels)
        y_ax.append(acc)
    plt.plot(x_ax,y_ax,'g-')
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Max_Depth")
    plt.show()

# Uncomment the following code to run 3.4
#dt_spam_test_predictions = useDecisionTree('spam')
#dt_titanic_test_predictions = useDecisionTree('titanic')
#rf_spam_test_predictions = useRandomForest('spam')
#rf_titanic_test_predictions = useRandomForest('titanic')

#results_to_csv(rf_spam_test_predictions, 'spam.csv')
#results_to_csv(rf_spam_test_predictions, 'titanic.csv')

#rf_titanic_test_predictions = rfTitanic()
#results_to_csv(rf_titanic_test_predictions, 'titanic.csv')

#Uncomment the following code to run 3.5

#plotValAccuracies()
