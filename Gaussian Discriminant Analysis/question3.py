import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
import math

def getPDF(mu, sigma):
    """returns X, Y, and the pdf of the Gaussian distribution with mean mu and covariance sigma"""
    eigs = np.linalg.eig(sigma)[0]
    std_devs = [math.sqrt(eig) for eig in eigs]
    x = np.linspace(mu[0]-3*std_devs[0], mu[0]+3*std_devs[0], 100)
    y = np.linspace(mu[1]-3*std_devs[1], mu[1]+3*std_devs[1], 100)
    X,Y = np.meshgrid(x, y)
    pos = np.dstack((X,Y))
    rv = multivariate_normal(mean=mu, cov=sigma)
    return X, Y, rv.pdf(pos)

def getDifInPDF(mu1, mu2, sigma1, sigma2):
    """returns X, Y, and the differnece between the pdfs of the Gaussian distributions with means mu1, mu2 and covariances sigma1, sigma2"""
    eigs1 = np.linalg.eig(sigma1)[0]
    eigs2 = np.linalg.eig(sigma2)[0]
    std_devs1 = [math.sqrt(eig) for eig in eigs1]
    std_devs2 = [math.sqrt(eig) for eig in eigs2]
    min_x = min(mu1[0]-3*std_devs1[0], mu2[0]-3*std_devs2[0])
    max_x = max(mu1[0]+3*std_devs1[0], mu2[0]+3*std_devs2[0])
    min_y = min(mu1[1]-3*std_devs1[1], mu2[1]-3*std_devs2[1])
    max_y = max(mu1[1]+3*std_devs1[1], mu2[1]+3*std_devs2[1])
    x = np.linspace(min_x, max_x, 100)
    y = np.linspace(min_y, max_y, 100)
    X,Y = np.meshgrid(x, y)
    pos = np.dstack((X,Y))
    rv1 = multivariate_normal(mean=mu1, cov=sigma1)
    rv2 = multivariate_normal(mean=mu2, cov=sigma2)
    return X, Y, rv1.pdf(pos)-rv2.pdf(pos)

def plot1(mu, sigma, part_num):
    """Plots the isocontour of the pdf of a Gaussian distribution with mean mu and covariance sigma.
    Used for parts 1 and 2"""
    X, Y, pdf = getPDF(mu, sigma)
    fig, ax = plt.subplots()
    ax.set_aspect(1)
    cs = ax.contour(X, Y, pdf)
    ax.clabel(cs, inline=1, fontsize=15)
    plt.title("Question 3 - Part " + str(part_num))
    #plt.contourf(X, Y, rv.pdf(pos), 10)
    plt.show()

def plot2(mu1, mu2, sigma1, sigma2, part_num) :
    """Plots the isocontour of the pdf of a Gaussian distribution (mu1, sigma1) - the pdf of the Gaussian distr (mu2, sigma2).
    X ~ N(m1,s1), Y ~ N(m2, s2), X-Y ~ N(m1-m2,s1+s2)
    Used for parts 3-5"""
    X, Y, dif = getDifInPDF(mu1, mu2, sigma1, sigma2)
    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, dif)
    ax.set_aspect(1)
    ax.clabel(cs, inline=1, fontsize=15)
    plt.title("Question 3 - Part " + str(part_num))
    #plt.contourf(X, Y, rv.pdf(pos), 10)
    plt.show()

# Part 1 ///////////////////////////////////////////////////////////////////////
def part1():
    mu = np.array([1,1])
    sigma = np.array([[1,0],[0,2]])
    plot1(mu, sigma, 1)
# Part 2 ///////////////////////////////////////////////////////////////////////
def part2():
    mu = np.array([-1,2])
    sigma = np.array([[2,1],[1,4]])
    plot1(mu, sigma, 2)
# Part 3 ///////////////////////////////////////////////////////////////////////
def part3():
    mu1 = np.array([0,2])
    mu2 = np.array([2,0])
    sigma1 = np.array([[2,1],[1,1]])
    sigma2 = np.array([[2,1],[1,1]])
    plot2(mu1,mu2,sigma1,sigma2, 3)
# Part 4 ///////////////////////////////////////////////////////////////////////
def part4():
    mu1 = np.array([0,2])
    mu2 = np.array([2,0])
    sigma1 = np.array([[2,1],[1,1]])
    sigma2 = np.array([[2,1],[1,4]])
    plot2(mu1,mu2,sigma1,sigma2, 4)
# Part 5 ///////////////////////////////////////////////////////////////////////
def part5():
    mu1 = np.array([1,1])
    mu2 = np.array([-1,-1])
    sigma1 = np.array([[2,0],[0,1]])
    sigma2 = np.array([[2,1],[1,2]])
    plot2(mu1,mu2,sigma1,sigma2, 5)


"""Uncomment one of the following lines of code in order to run the code for the associated part"""

part1()
part2()
part3()
part4()
part5()