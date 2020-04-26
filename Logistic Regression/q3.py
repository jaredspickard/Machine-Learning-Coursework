import scipy.io
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
from save_csv import results_to_csv
from scipy.special import expit

np.random.seed(123987)

wine_data = scipy.io.loadmat("data.mat")
data_not_norm, labels = shuffle(wine_data["X"], wine_data["y"])
test_data = wine_data["X_test"]

def normalizeFeatures(X, mns, stds):
    """Normalizes the columns of X and appends a column of 1's at the end"""
    new_arr = []
    for i in range(len(X.T)):
        col = X.T[i]
        mean = mns[i]
        std_dev = stds[i]
        means = np.array([mean for _ in col])
        new_col = (col - means)
        if std_dev > 0:
            new_col /= std_dev
        new_arr.append(new_col)
    ones = np.ones((len(X),1))
    return np.hstack((np.array(new_arr).T, ones))

def calcMeansAndStdDevs(data):
    """Returns a list of the means of each columns and the std devs of each column of the data"""
    means, std_devs = [], []
    for col in data.T:
        means.append(np.mean(col))
        std_devs.append(np.std(col))
    return means, std_devs

#calculate the means and std devs of each column of the training data
means, std_devs = calcMeansAndStdDevs(data_not_norm)
#whiten the data and test data using the values above
data = normalizeFeatures(data_not_norm, means, std_devs)
test_data = normalizeFeatures(test_data, means, std_devs)
data_not_norm = np.hstack((data_not_norm, np.ones((len(data_not_norm),1))))
#split the training and validation data/labels
training_data = data[:5000]
val_data = data[5000:]
training_labels = labels[:5000]
val_labels = labels[5000:]

def J(X, y, w, lamb):
    """The cost function for l2 regularized logistic regression"""
    summation = 0
    for i in range(len(X)):
        s_val = s_stoch(X[i], w)
        summation += y[i]*np.log(np.clip(s_val, 0.00001, 1-0.00001)) + (1-y[i])*np.log(1-np.clip(s_val, 0.00001, 1-0.00001))
    return -1/len(X)*summation[0] + lamb*np.linalg.norm(w)

def s(X,w):
    """Calculates the logistic function on each value in the array X.w"""
    gammas = np.dot(X,w)
    return np.array([np.where(expit(g) > 0.5, 1, 0) for g in gammas])

def s_stoch(xi, w):
    g = np.dot(xi, w)
    return np.where(expit(g) > 0.5, 1, 0)

# Initialize random weight vectors
w_bgd = np.random.rand(len(training_data[0]),1)
w_sgd = np.random.rand(len(training_data[0]))
w_dsgd = np.copy(w_sgd)

num_iters = [100, 250, 500, 1000, 2500, 5000, 7500, 10000, 20000]
dif_iters = [100, 150, 250, 500, 1500, 2500, 2500, 2500, 10000]

# we will be running gradient descent on each value in dif_iters (not resetting the weight between iterations)
# this is equivalent to running gd with num_iters iterations, but resetting the weights each iteration

# Part 2 ////////////////////////////////////////////////////////////////////////////////////////////////////////
def batchGradientDescent(X, y, w, lamb, eps, num_iters):
    """Runs BGD for at most num_iters iterations, returning the trained weight vector"""
    for it in range(num_iters):
        update = eps*(np.dot(X.T,y-s(X,w)) - lamb*w)
        w = w + update
    return w

# Choosing parameters for BGD
best_params_bgd = (0,0)
best_acc_bgd = 0
for reg in [0.1, 0.01, 0.001, 0.0001, 0]:
    for step in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]:
        w = np.copy(w_bgd)
        w = batchGradientDescent(training_data, training_labels, w, reg, step, 5000)
        preds = s(val_data, w)
        acc = accuracy_score(preds, val_labels)
        if acc > best_acc_bgd:
            best_acc_bgd = acc
            best_params_bgd = (reg, step)
print("Best parameters for BGD (reg, step):", best_params_bgd)
print("In 5000 iterations of BGD, these achieved an accuracy of", best_acc_bgd)

bgd_regularizer = 0.0001
bgd_step_size = 0.00001

bgd_costs = []
for it in dif_iters:
    print(it)
    w_bgd = batchGradientDescent(data, labels, w_bgd, bgd_regularizer, bgd_step_size, it)
    J_val = J(data, labels, w_bgd, 0)
    print(J_val)
    bgd_costs.append(J_val)
plt.plot(num_iters,bgd_costs,'ro-')
plt.ylabel("Cost Function Value")
plt.xlabel("Number of Training Iterations")
plt.title("Cost Value vs Training Iterations for Batch Gradient Descent")
plt.show()

# Part 4 ////////////////////////////////////////////////////////////////////////////////////////////////////////
def stochasticGradientDescent(X, y, w, lamb, eps, num_iters):
    """Runs SGD for at most num_iters iterations, returning the trained weight vector"""
    for it in range(num_iters):
        ind = np.random.randint(0, len(X))
        xi = X[ind]
        yi = y[ind]
        update = eps*((yi-s_stoch(xi,w))*xi - lamb*w)
        w = w + update
    return w

# Choosing parameters for SGD
best_params_sgd = (0,0)
best_acc_sgd = 0
for reg in [0.1, 0.01, 0.001, 0.0001, 0]:
    for step in [0.001, 0.01, 0.1, 0.5]:
        w = np.copy(w_sgd)
        w = stochasticGradientDescent(training_data, training_labels, w, reg, step, 15000)
        preds = s(val_data, w)
        acc = accuracy_score(preds, val_labels)
        if acc > best_acc_sgd:
            best_acc_sgd = acc
            best_params_sgd = (reg, step)
print("Best parameters for SGD (reg, step):", best_params_sgd)
print("In 15000 iterations of SGD, these achieved an accuracy of", best_acc_sgd)

sgd_regularizer = 0.001
sgd_step_size = 0.01

sgd_costs = []
for it in dif_iters:
    print(it)
    w_sgd = stochasticGradientDescent(data, labels, w_sgd, sgd_regularizer, sgd_step_size, it)
    J_val = J(data, labels, w_sgd, 0)
    print(J_val)
    sgd_costs.append(J_val)
plt.plot(num_iters,sgd_costs,'ro-')
plt.ylabel("Cost Function Value")
plt.xlabel("Number of Training Iterations")
plt.title("Cost Value vs Training Iterations for Stochastic Gradient Descent")
plt.show()

# Part 5 ////////////////////////////////////////////////////////////////////////////////////////////////////////

def dynamicStochasticGradientDescent(X, y, w, lamb, delta, num_iters):
    """Runs SGD for at most num_iters iterations, returning the trained weight vector"""
    for it in range(num_iters):
        ind = np.random.randint(0, len(X))
        xi = X[ind]
        yi = y[ind]
        update = (delta/it)*((yi-s_stoch(xi,w))*xi - lamb*w)
        w = w + update
    return w

dsgd_costs = []
for it in dif_iters:
    print(it)
    w_dsgd = stochasticGradientDescent(data, labels, w_dsgd, sgd_regularizer, sgd_step_size, it)
    J_val = J(data, labels, w_dsgd, 0)
    print(J_val)
    dsgd_costs.append(J_val)
plt.plot(num_iters,dsgd_costs,'ro-')
plt.ylabel("Cost Function Value")
plt.xlabel("Number of Training Iterations")
plt.title("Cost Value vs Training Iterations for Dynamic Stochastic Gradient Descent")
plt.show()

# Part 6 ////////////////////////////////////////////////////////////////////////////////////////////////////////
# train the model and run it on the test data

w = batchGradientDescent(data, labels, w_bgd, bgd_regularizer, bgd_step_size, 50)
preds = s(test_data, w)
results_to_csv(preds[:,0])