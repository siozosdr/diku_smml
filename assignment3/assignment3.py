from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import csv
from NN_ import Neuron_Network

def read_sinc_files(filename):
    with open(filename) as rows:
        xs = []
        ys = []
        for row in csv.reader(rows, delimiter = ' '):
            xs.append(float(row[0]))
            ys.append(float(row[1]))
        return np.array([xs]).T, np.array([ys]).T


def normalize_park(data, mu, sigma):
    # normalizes the parkinson data
    for row in range(len(data)):
        for col in range(len(data[row])):
            if col != len(data[row]) -1:
                data[row][col] = (data[row][col] - mu[col]) / np.sqrt(sigma[col])

def normalize_sinc(xs, mu, sigma):
    for row in range(len(xs)):
        xs[row] = (xs[row] - mu) / np.sqrt(sigma)

def generate_data():
    xs = np.arange(-5,5,0.1)
    ys = [np.sin(x)/x for x in xs]
    return np.array([xs]).T, np.array([ys]).T

def iii_1_1():

    xs, ys = read_sinc_files('./sincTrain25.dt')
    vx, vy = read_sinc_files(('./sincValidate10.dt'))
    # Normalize?
    # xs_mu = np.mean(xs)
    # xs_sigma = np.var(xs)
    # ys_mu = np.mean(ys)
    # ys_sigma = np.var(ys)
    # normalize_sinc(xs, xs_mu, xs_sigma)
    # normalize_sinc(ys, ys_mu, ys_sigma)
    no_epochs = 5000
    NN = Neuron_Network(1, 2, 1)
    _, c1 = NN.train(xs, ys, no_epochs)
    NN2 = Neuron_Network(1, 20, 1)
    _, c2 = NN2.train(xs, ys, no_epochs)
    y1 = NN.forward(xs)
    y2 = NN2.forward(xs)
    # plot of cost basedo n epoch of learning
    epochs = [i for i in range(len(c1))]
    plt.plot(epochs, c1, 'r')
    epochs = [i for i in range(len(c2))]
    plt.plot(epochs, c2, 'g')
    plt.show()
    # plot predictions of the 2 NNs
    plt.plot(xs, y1, 'r.')
    plt.plot(xs, y2, 'g.')
    plt.plot(xs, ys, 'bo')
    plt.show()
    # plot of validation data
    vy1 = NN.forward(vx)
    vy2 = NN2.forward(vy)
    plt.plot(vx, vy1, 'r.')
    plt.plot(vx, vy2, 'g.')
    plt.plot(vx, vy, 'b+')
    plt.show()
    # plot for (-10, 10)
    testxs = np.array([i/100 for i in range(-1000, 1000, 5)]).reshape(2000/5, 1)
    testys = NN.forward(testxs)
    plt.plot(testxs, testys, 'r.')
    plt.show()


def function_1_1(xs):
    result = []
    for x in xs:
        temp = np.sin(x)/x
        result.append(temp)
    result = np.array(result)
    result = result.reshape((len(result),1))
    return result


def iii_1_2():
    xs, ys = read_sinc_files('./sincTrain25.dt')
    vx, vy = read_sinc_files(('./sincValidate10.dt'))
    # Normalize?
    # xs_mu = np.mean(xs)
    # xs_sigma = np.var(xs)
    # ys_mu = np.mean(ys)
    # ys_sigma = np.var(ys)
    # normalize_sinc(xs, xs_mu, xs_sigma)
    # normalize_sinc(ys, ys_mu, ys_sigma)
    no_epochs = 1000
    NN = Neuron_Network(1, 2, 1)
    _, c1 = NN.train(xs, ys, no_epochs)
    NN2 = Neuron_Network(1, 20, 1)
    _, c2 = NN2.train(xs, ys, no_epochs)
    y1 = NN.forward(xs)
    y2 = NN2.forward(xs)
    # plot of cost basedo n epoch of learning
    epochs = [i for i in range(len(c1))]
    plt.plot(epochs, c1, 'r')
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.savefig('epochs-cost1.png')
    plt.show()

    epochs = [i for i in range(len(c2))]
    plt.plot(epochs, c2, 'g')
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.savefig('epochs-cost2.png')
    plt.show()
    # plot predictions of the 2 NNs
    plt.plot(xs, y1, 'r.')
    plt.plot(xs, y2, 'g.')
    plt.plot(xs, ys, 'bo')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('predictions.png')
    plt.show()
    # plot of validation data
    vy1 = NN.forward(vx)
    vy2 = NN2.forward(vy)
    plt.plot(vx, vy1, 'r.')
    plt.plot(vx, vy2, 'g.')
    plt.plot(vx, vy, 'b+')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.savefig('predictions_from_validation.png')
    # plot for (-10, 10)
    testxs = np.array([i/100 for i in range(-1000, 1000, 5)]).reshape(2000/5, 1)
    testys = NN.forward(testxs)
    plt.plot(testxs, testys, 'r.')
    ys = function_1_1(testxs)
    plt.plot(testxs, ys, 'b.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('function1_1_and_prediction.png')
    plt.show()

def iii_2_1():

    park_train = np.loadtxt('./parkinsonsTrainStatML.dt')
    park_test = np.loadtxt('./parkinsonsTestStatML.dt')

    # find the mean and sigma for the data
    train_mu = np.mean(park_train, axis=0)
    train_sigma = np.var(park_train, axis=0)

    # report it
    print '-- Mean and train_sigma, respectively, for training features --'
    for m in range(len(train_mu)):
        print '\tFeature {0}: {1:.3f}, {2:.3f}'.format(m+1, train_mu[m], train_sigma[m])

    # normalize it
    normalize_park(park_train, train_mu, train_sigma)
    train_mu_norm = np.mean(park_train, axis=0)
    train_sigma_norm = np.var(park_train, axis=0)

    # check the normalization. Mean should be close to zero and variance close to one.
    print '-- Normalized mean and sigma, respectively, for training features --'
    for m in range(len(train_mu_norm)):
        print '\tFeature {0}: {1:.3f}, {2:.3f}'.format(m+1, train_mu_norm[m], train_sigma_norm[m])

    # finally, normalize the test data with the calculated mean and variance for the training set
    normalize_park(park_test, train_mu, train_sigma)

    test_mu_norm = np.mean(park_test, axis=0)
    test_sigma_norm = np.var(park_test, axis=0)

    # report it
    print '-- Normalized mean and sigma, respectively, for testing features --'
    for m in range(len(test_mu_norm)):
        print '\tFeature {0}: {1:.3f}, {2:.3f}'.format(m+1, test_mu_norm[m], test_sigma_norm[m])

    # finally, write it to file

    f = open('./normalizedConvertedTest', 'w')
    for t in park_test:
        f.write(str(int(t[-1])) + ' ')
        for i in range(0,len(t)-1):
            f.write(str(i) + ':' + str(t[i]) + ' ')
        f.write('\n')
    f.close()

if __name__ == '__main__':
    # iii_1_1()
    iii_1_2()
    # iii_2_1()






