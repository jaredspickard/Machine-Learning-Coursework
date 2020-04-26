import scipy.io
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from save_csv import results_to_csv

#============================ Question 1 =============================
"""import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io
for data_name in ["mnist", "spam", "cifar10"]:
    data = io.loadmat("data/%s_data.mat" % data_name)
    print("\nloaded %s data!" % data_name)
    fields = "test_data", "training_data", "training_labels"
    for field in fields:
        print(field, data[field].shape)"""

#============================ Question 2 =============================
# 2a) ---------
"""Each of these functions returns a dictionarywith the following fields: 'training_data', 'training_labels', 'validation_data', 'validation_labels'"""
def split_mnist_data():
    mnist_data = scipy.io.loadmat('data/mnist_data.mat')
    data, labels = shuffle(mnist_data["training_data"], mnist_data["training_labels"])
    return split_data(data, labels, 10000)
# 2b) ---------
def split_spam_data():
    spam_data = scipy.io.loadmat('data/spam_data.mat')
    data, labels = shuffle(spam_data["training_data"], spam_data["training_labels"])
    val_size = int(len(data)*.2)
    return split_data(data, labels, val_size)
# 2c) ---------
def split_cifar10_data():
    cifar10_data = scipy.io.loadmat('data/cifar10_data.mat')
    data, labels = shuffle(cifar10_data["training_data"], cifar10_data["training_labels"])
    return split_data(data, labels, 5000)
# helper function --------
def split_data(data, labels, val_size):
    ret_data = dict()
    ret_data["training_data"] = data[val_size:]
    ret_data["training_labels"] = labels[val_size:]
    ret_data["validation_data"] = data[:val_size]
    ret_data["validation_labels"] = labels[:val_size]
    return ret_data

#============================ Question 3 =============================

def train_mnist():
    data = split_mnist_data()
    return train(data, [100, 200, 500, 1000, 2000, 5000, 10000])

def train_spam():
    data = split_spam_data()
    return train(data, [100, 200, 500, 1000, 2000, len(data["training_data"])])

def train_cifar10():
    data = split_cifar10_data()
    return train(data, [100, 200, 500, 1000, 2000, 5000])

def train(data, examples):
    """Trains a linear svm model using each number in the list 'examples' as the number of training examples.
    Returns the list of error rates and number of examples"""
    error_rate = []
    for num_ex in examples:
        training_data = data["training_data"][:num_ex]
        training_labels = data["training_labels"][:num_ex]
        clf = svm.SVC(kernel='linear')
        clf.fit(training_data, training_labels)
        val_true = data["validation_labels"]
        val_pred = clf.predict(data["validation_data"])
        accuracy = accuracy_score(val_true, val_pred)
        error_rate.append(1-accuracy)
    return examples, error_rate

def plot_error_rates(x, y):
    plt.plot(x,y,'g^')
    plt.ylabel("Error Rates")
    plt.xlabel("Number of Training Examples")
    plt.show()

#x, y = train_mnist()
#x, y = train_spam()
#x, y = train_cifar10()
#plot_error_rates(x,y)

#============================ Question 4 =============================

def mnist_hyperparameter():
    """returns the best value of C"""
    data = split_mnist_data()
    training_data = data["training_data"][:10000]
    training_labels = data["training_labels"][:10000]
    val_true = data["validation_labels"]
    C_vals = []
    accuracies = []
    for exp in range(-8,-1):
        C = 10**(exp)
        C_vals.append(C)
        clf = svm.SVC(kernel='linear',C=C)
        clf.fit(training_data, training_labels)
        val_pred = clf.predict(data["validation_data"])
        accuracy = accuracy_score(val_true, val_pred)
        accuracies.append(accuracy)
    top_acc = 0
    top_ind = 0
    top_C = 0
    for ind in range(len(accuracies)):
        if accuracies[ind] > top_acc:
            top_ind = ind
            top_acc = accuracies[ind]
            top_C = C_vals[ind]
    #print("Top C:", top_C)
    mnist_data = scipy.io.loadmat('data/mnist_data.mat')
    test_data = mnist_data["test_data"]
    training_data, training_labels = shuffle(mnist_data["training_data"], mnist_data["training_labels"])
    clf = svm.LinearSVC(C=top_C)
    clf.fit(training_data, training_labels)
    test_preds = clf.predict(test_data)
    results_to_csv(test_preds)
    return C_vals, accuracies

#c, acc = mnist_hyperparameter()
#print(c)
#print(acc)

#============================ Question 5 =============================

def spam_hyperparameter():
    C_vals = []
    accuracies = []
    data = split_spam_data_crossval(5)
    for exp in range(-5,4):
        print(exp)
        C = 10**exp
        C_vals.append(C)
        accuracy = 0
        for k in range(5):
            training_data = []
            training_labels = []
            for i in range(5):
                if not i == k:
                    training_data.append(data[i]["data"])
                    training_labels.append(data[i]["labels"])
            val_data = data[k]["data"]
            val_true = data[k]["labels"]
            clf = svm.LinearSVC(C=C)
            for ind in range(len(training_data)):
                clf.fit(training_data[ind], training_labels[ind])
            val_pred = clf.predict(val_data)
            accuracy += accuracy_score(val_true, val_pred)
        accuracies.append(accuracy/5)
    top_acc = 0
    top_ind = 0
    top_C = 0
    for ind in range(len(accuracies)):
        if accuracies[ind] > top_acc:
            top_ind = ind
            top_acc = accuracies[ind]
            top_C = C_vals[ind]
    spam_data = scipy.io.loadmat('data/spam_data.mat')
    test_data = spam_data["test_data"]
    training_data, training_labels = shuffle(spam_data["training_data"], spam_data["training_labels"])
    clf = svm.LinearSVC(C=top_C)
    clf.fit(training_data, training_labels)
    test_preds = clf.predict(test_data)
    results_to_csv(test_preds)
    return C_vals, accuracies

def split_spam_data_crossval(num_sets):
    """returns num_sets disjoint sets to be used for k-fold cross-validation"""
    spam_data = scipy.io.loadmat('data/spam_data.mat')
    data, labels = shuffle(spam_data["training_data"], spam_data["training_labels"])
    val_size = int(len(data)/num_sets)
    sets = []
    for num in range(num_sets-1):
        ret_data = dict()
        ret_data["data"] = data[num*val_size:(num+1)*val_size]
        ret_data["labels"] = labels[num*val_size:(num+1)*val_size]
        sets.append(ret_data)
    ret_data = dict()
    ret_data["data"] = data[(num+1)*val_size:]
    ret_data["labels"] = labels[(num+1)*val_size:]
    sets.append(ret_data)
    return sets

#C, acc = mnist_hyperparameter()
#print(C)
#print(acc)

#============================ Question 6 =============================

def cifar10_hyperparameter():
    """returns the best value of C"""
    data = split_cifar10_data()
    training_data = data["training_data"][:2000]
    training_labels = data["training_labels"][:2000]
    val_true = data["validation_labels"]
    C_vals = []
    accuracies = []
    for exp in range(-2,2):
        print(exp)
        C = 10**(exp)
        C_vals.append(C)
        clf = svm.LinearSVC(C=C)
        clf.fit(training_data, training_labels)
        val_pred = clf.predict(data["validation_data"])
        accuracy = accuracy_score(val_true, val_pred)
        accuracies.append(accuracy)
    top_acc = 0
    top_ind = 0
    top_C = 0
    for ind in range(len(accuracies)):
        if accuracies[ind] > top_acc:
            top_ind = ind
            top_acc = accuracies[ind]
            top_C = C_vals[ind]
    cifar10_data = scipy.io.loadmat('data/cifar10_data.mat')
    test_data = cifar10_data["test_data"]
    training_data, training_labels = shuffle(cifar10_data["training_data"], cifar10_data["training_labels"])
    clf = svm.LinearSVC(C=top_C)
    clf.fit(training_data, training_labels)
    test_preds = clf.predict(test_data)
    results_to_csv(test_preds)
    return C_vals, accuracies

#c, acc = cifar10_hyperparameter()
#print(c)
#print(acc)