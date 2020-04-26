import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math

np.random.seed(123456) # initialize seed so all results are reproducible

X1 = np.random.normal(3,3, 100) #100 samples for X1
early_X2 = np.random.normal(4,2,100) #100 samples to be used to calculate X2
X2 = [0.5*X1[i] + early_X2[i] for i in range(len(X1))]

# Part a /////////////////////////////////////////////////////////////////////////
mean = np.array([np.mean(X1), np.mean(X2)])
print("Mean: ", mean)
print()

# Part b ////////////////////////////////////////////////////////////////////////
covariance_matrix = np.cov([X1,X2])
print("Covariance Matrix:\n", covariance_matrix)
print()

# Part c ////////////////////////////////////////////////////////////////////////
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
print("Eigenvalues: ", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
print()

# Part d ///////////////////////////////////////////////////////////////////////
def partD():
    fig, ax = plt.subplots()
    ax.scatter(X1,X2)
    ax.set_aspect(1)
    alpha1 = math.sqrt(eigenvalues[0])/math.sqrt(eigenvectors[0][0]**2 + eigenvectors[1][0]**2)
    alpha2 = math.sqrt(eigenvalues[1])/math.sqrt(eigenvectors[0][1]**2 + eigenvectors[1][1]**2)
    plt.arrow(mean[0], mean[1], alpha1*(eigenvectors[0][0]), alpha1*(eigenvectors[1][0]))
    plt.arrow(mean[0], mean[1], alpha2*(eigenvectors[0][1]), alpha2*(eigenvectors[1][1]))
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Distribution of 100 Random Sample Points")
    plt.xlim(-15,15)
    plt.ylim(-15,15)
    plt.show()

"Uncomment the following line to run part D"
partD()

# Part e ///////////////////////////////////////////////////////////////////////
def partE():
    uT = eigenvectors.T
    X_rot1 = []
    X_rot2 = []
    for i in range(len(X1)):
        point = np.array([X1[i],X2[i]])
        holder = uT*(point-mean)
        X_rot1.append(holder[0])
        X_rot2.append(holder[1])
    fig, ax = plt.subplots()
    ax.scatter(X_rot1,X_rot2)
    ax.set_aspect(1)
    plt.title("100 Random Sample Points, Centered and Rotated by UT")
    plt.xlim(-15,15)
    plt.ylim(-15,15)
    plt.xlabel("Eigenvector 1")
    plt.ylabel("Eigenvector 2")
    plt.show()

"Uncomment the following line to run part E"
partE()
