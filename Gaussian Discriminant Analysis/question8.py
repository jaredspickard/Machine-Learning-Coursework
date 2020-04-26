import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle # a TA on piazza said we could use this function!
import matplotlib.pyplot as plt
from collections import Counter
from save_csv import results_to_csv


# Part 1 ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def getGaussDistr():
    """Returns the Mean vectors and Covariance matrices for the mnist data set.
    Means[i] is the mean vector for the class of digit i.
    Covs[i] is the Covariance vector for the class of digit i."""
    mnist_data = scipy.io.loadmat("../data/mnist_data.mat")
    training_data, training_labels = shuffle(mnist_data["training_data"], mnist_data["training_labels"])
    Means = calculateMeans(training_data, training_labels)[0]
    Covs = calculateCovs(training_data, training_labels)
    return Means, Covs

def calculateMeans(training_data, training_labels):
    """Returns Means, priors. 
    Means[i] is the mean vector for the class of digit i
    priors[i] is the prior for the class of digit i"""
    sum_matrix = [np.array([]) for _ in range(10)] #this will hold an array of sums of the data (to be used to calculate means)
    counts = [0 for _ in range(10)] #this will be used to hold counts of the data (also for means)
    for i in range(len(training_labels)):
        index = training_labels[i][0] #this number tells us which class this data point belongs to 
        img = normalize(training_data[i]) #we set img to be the normalized data point
        if len(sum_matrix[index]) == 0: #first time encountering this class
            sum_matrix[index] = img
        else:
            sum_matrix[index] += img
        counts[index] += 1 #increment the number of times we've encountered a data point belonging to this class
    Means = [sum_matrix[i]/counts[i] for i in range(10)] #Means[i] = mean of class i
    return Means, np.array(counts)*sum(counts)

def calculateCovs(training_data, training_labels):
    """Returns Covs, where Covs[i] is the Covariance vector for the class of digit i"""
    Cov_data = [[] for _ in range(10)]
    for i in range(len(training_labels)):
        index = training_labels[i][0] #which class the data point belongs to
        img = normalize(training_data[i]) #normalize the data point
        img = addGaussianNoise(img) #adds guassian noise so the covariance matrix isn't singular
        Cov_data[index].append(img) #Cov_data[index] holds data to calculate the covariance of the index class
    Covs = []
    for i in range(len(Cov_data)):
        c = np.array(Cov_data[i])
        Covs.append(np.cov(c.T))
    return Covs

# Part 2 ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def visualizeCovariance(digit, Covs):
    """Plots a visualization of the covariance matrix for digit (Cov[digit])"""
    cov = Covs[digit]
    plt.imshow(cov)
    plt.title("Visualization of the Covariance Matrix for the Digit " + str(digit))
    plt.show()

# Part 3 ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Part a ------------------------------------------------------------

def LDA():
    """Classify the test data points using LDA"""
    training_data, training_labels, validation_data, validation_labels = train_val_split("../data/mnist_data.mat", 10000)
    sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
    error_rates = []
    for size in sizes:
        Means, Cov, priors = trainLDA(training_data, training_labels, size)
        error_rates.append(getLDAErrorRate(Means, Cov, priors, validation_data, validation_labels))
    plotErrorRates(sizes, error_rates, "LDA Error Rates")

def trainLDA(training_data, training_labels, size):
    """Trains our LDA function using 'size' training points, returns Means, Covariance Matrix, and Priors"""
    t_data, t_labels = shuffle(training_data, training_labels)
    data = t_data[:size]
    labels = t_labels[:size]
    Means, priors = calculateMeans(data, labels)
    Covs = calculateCovs(data, labels) #holds individual covariances
    cov_sum = np.zeros(shape=(784,784)) #will hold the sum of all covariances
    for c in Covs:
        cov_sum += c
    Cov = .1*cov_sum
    return Means, Cov, priors

def getLDAErrorRate(Means, Cov, priors, validation_data, validation_labels):
    """Returns the error rate of our LDA model (mu[i] = Means[i], Sigma = Cov)"""
    z = [] #list of values representing Sigma^(-1)*Mu[class]
    for i in range(10):
        z.append(np.linalg.solve(Cov, Means[i]))
    predictions = getLDAPredictions(Means, Cov, priors, validation_data, validation_labels, z)
    num_correct = 0 #number of correct predicitions
    for i in range(len(validation_data)):
        if predictions[i] == validation_labels[i]:
            num_correct += 1
    return 1 - num_correct/len(validation_data)

def getLDAPredictions(Means, Cov, priors, validation_data, validation_labels, z):
    """Returns a list of predictions of our LDA model"""
    predictions = []
    for data_point in validation_data:
        max_class = 0
        max_val = -1*float('inf')
        for digit in range(10):
            val = np.dot(z[digit], data_point) - 0.5* np.dot(z[digit], Means[digit]) + np.log(priors[digit])
            if val > max_val:
                max_val = val
                max_class = digit
        predictions.append(max_class)
    return predictions


# Part b ------------------------------------------------------------
def QDA():
    """Classify the test data points using QDA"""
    training_data, training_labels, validation_data, validation_labels = train_val_split("../data/mnist_data.mat", 10000)
    sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
    error_rates = []
    for size in sizes:
        Means, Covs, priors = trainQDA(training_data, training_labels, size)
        error_rates.append(getQDAErrorRate(Means, Covs, priors, validation_data, validation_labels))
    plotErrorRates(sizes, error_rates, "QDA Error Rates")

def trainQDA(training_data, training_labels, size):
    """Trains our QDA function using 'size' training points, returns Means, Covariance Matrices, and Priors"""
    t_data, t_labels = shuffle(training_data, training_labels)
    data = t_data[:size]
    labels = t_labels[:size]
    Means, priors = calculateMeans(data, labels)
    Covs = calculateCovs(data, labels)
    return Means, Covs, priors

def getQDAErrorRate(Means, Covs, priors, validation_data, validation_labels):
    """Returns the error rate of our QDA model (mu[i] = Means[i], Sigma = Cov)"""
    predictions = getQDAPredictions(Means, Covs, priors, validation_data)
    num_correct = 0 #number of correct predicitions
    for i in range(len(validation_data)):
        if predictions[i] == validation_labels[i]:
            num_correct += 1
    return 1 - num_correct/len(validation_data)

def getQDAPredictions(Means, Covs, priors, validation_data):
    """Returns a list of predictions of our QDA model"""
    predictions = []
    determinants = [np.linalg.slogdet(Covs[i]) for i in range(10)]
    inverses = [np.linalg.inv(Covs[i]) for i in range(10)]
    for i in range(len(validation_data)):
        data_point = validation_data[i]
        max_class = 0
        max_val = -1*float('inf')
        for digit in range(10):
            z_c = np.dot(inverses[digit], data_point-Means[digit])
            val = -0.5*np.dot(z_c, data_point-Means[digit]) - 0.5*determinants[digit][1] + np.log(priors[digit])
            if val > max_val:
                max_val = val
                max_class = digit
        predictions.append(max_class)
    return predictions

# Part d ------------------------------------------------------------
def testQDA():
    """trains our QDA model on 30000 points of training data and returns our predictions for the test data"""
    mnist_data = scipy.io.loadmat("../data/mnist_data.mat")
    training_data, training_labels = shuffle(mnist_data["training_data"], mnist_data["training_labels"])
    training_data = training_data[:30000]
    training_labels = training_labels[:30000]
    test_data = mnist_data["test_data"]
    Means, priors = calculateMeans(training_data, training_labels)
    Covs = calculateCovs(training_data, training_labels)
    predictions = getQDAPredictions(Means, Covs, priors, test_data)
    return predictions

def QDAIndividualErrors():
    training_data, training_labels, validation_data, validation_labels = train_val_split("../data/mnist_data.mat", 10000)
    sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
    error_rates = []
    for size in sizes:
        Means, Covs, priors = trainQDA(training_data, training_labels, size)
        error_rates.append(getIndividualErrorRates(Means, Covs, priors, validation_data, validation_labels))
        print(error_rates)
    error_rates = np.array(error_rates)
    plotIndividualErrorRates(error_rates.T)

def getIndividualErrorRates(Means, Covs, priors, validation_data, validation_labels):
    predictions = getQDAPredictions(Means, Covs, priors, validation_data)
    num_error = [0 for _ in range(10)] #value at each index represents number of incorrect predicitions for that class
    num_total = [0 for _ in range(10)]
    for i in range(len(validation_data)):
        digit = validation_labels[i][0]
        if not predictions[i] == digit:
            num_error[digit] += 1
        num_total[digit] += 1
    error_rates = [num_error[digit]/num_total[digit] for digit in range(10)]
    return error_rates

def plotIndividualErrorRates(error_rates):
    print("wow")
    x_axis = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
    for i in range(len(error_rates)):
        plt.plot(x_axis, error_rates[i])
    plt.legend([i for i in range(10)], loc=1)
    plt.title("Error Rates for Individual Digits")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Error Rate")
    plt.show()

# Part 4 ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def spamQDA():
    """Runs QDA on spam data, returning the predictions"""
    spam_data = scipy.io.loadmat("../data/spam_data.mat")
    training_data, training_labels = shuffle(spam_data["training_data"], spam_data["training_labels"])
    test_data = spam_data["test_data"]
    Means, Covs, priors = trainSpamQDA(training_data, training_labels)
    predictions = getSpamQDAPredictions(Means, Covs, priors, test_data)
    return predictions

def trainSpamQDA(training_data, training_labels):
    Means, priors = calculateSpamMeans(training_data, training_labels)
    Covs = calculateSpamCovs(training_data, training_labels)
    return Means, Covs, priors

def calculateSpamMeans(training_data, training_labels):
    """Returns Means, priors. 
    Means[0] is the mean vector for the spam class, Means[1] for ham
    priors[0] is the prior for the spam class, priors[1] for ham"""
    sum_matrix = [np.array([]), np.array([])] #this will hold an array of sums of the data (to be used to calculate means)
    counts = [0, 0] #this will be used to hold counts of the data (also for means)
    for i in range(len(training_labels)):
        index = training_labels[i][0] #this number tells us which class this data point belongs to 
        img = normalize(training_data[i]) #we set img to be the normalized data point
        if len(sum_matrix[index]) == 0: #first time encountering this class
            sum_matrix[index] = img
        else:
            sum_matrix[index] += img
        counts[index] += 1 #increment the number of times we've encountered a data point belonging to this class
    Means = [sum_matrix[0]/counts[0], sum_matrix[1]/counts[1]] #Means[i] = mean of class i
    return Means, np.array(counts)*sum(counts)

def calculateSpamCovs(training_data, training_labels):
    """Returns Covs, where Covs[0] is the Covariance vector for the spam class and Covs[1] is for the ham class"""
    Cov_data = [[], []]
    for i in range(len(training_labels)):
        index = training_labels[i][0] #which class the data point belongs to
        img = normalize(training_data[i]) #normalize the data point
        img = addGaussianNoise(img) #adds guassian noise so the covariance matrix isn't singular
        Cov_data[index].append(img) #Cov_data[index] holds data to calculate the covariance of the index class
    Covs = []
    for i in range(len(Cov_data)):
        c = np.array(Cov_data[i])
        Covs.append(np.cov(c.T))
    return Covs

def getSpamQDAPredictions(Means, Covs, priors, validation_data):
    predictions = []
    determinants = [np.linalg.slogdet(Covs[0]), np.linalg.slogdet(Covs[1])]
    inverses = [np.linalg.inv(Covs[0]), np.linalg.inv(Covs[1])]
    for i in range(len(validation_data)):
        data_point = validation_data[i]
        max_class = 0
        max_val = -1*float('inf')
        for j in range(2):
            z_c = np.dot(inverses[j], data_point-Means[j])
            val = -0.5*np.dot(z_c, data_point-Means[j]) - 0.5*determinants[j][1] + np.log(priors[j])
            if val > max_val:
                max_val = val
                max_class = j
        predictions.append(max_class)
    return predictions

# Helper Functions /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def train_val_split(directory, split):
    """returns training_data, training_labels, validation_data, validation_labels, with the validation being of size split"""
    mat = scipy.io.loadmat(directory)
    data, labels = shuffle(mat["training_data"], mat["training_labels"])
    training_data = data[split:]
    training_labels = labels[split:]
    validation_data = data[:split]
    validation_labels = labels[:split]
    return training_data, training_labels, validation_data, validation_labels

def normalize(arr):
    """returns a normalized array"""
    norm = np.linalg.norm(arr)
    if not norm == 0:
        arr = (1/norm)*arr
    return arr

def plotErrorRates(x,y,title):
    """Plots x vs y with title"""
    plt.plot(x,y,'bo-')
    plt.ylabel("Error Rates")
    plt.xlabel("Number of Training Examples")
    plt.title(title)
    plt.show()

np.random.seed(123456)

def addGaussianNoise(m):
    noise = .01*np.random.normal(0,1,len(m))
    return m + noise

# Running the Code /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

"To run the code for the ith part of question 8, make sure that the line(s) of code beneath the comment 'Part i' are uncommented"

# Part 1
#means, covs = getGaussDistr()

# Part 2
#means, covs = getGaussDistr()
#visualizeCovariance(6, covs)

# Part 3a
LDA()

# Part 3b
QDA()

# Part 3d
#predictions = testQDA()
#results_to_csv(np.array(predictions))
#QDAIndividualErrors()

# Part 4
#spam_predictions = spamQDA()
#results_to_csv(np.array(spam_predictions))

"NOTES: subtract mean when normalizing??"