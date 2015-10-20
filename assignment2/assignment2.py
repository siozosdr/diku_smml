from __future__ import division
import matplotlib.pyplot as plt
import math
import numpy as np
import csv


def ii_1_1():
    # read in data
    train = read_file('./IrisTrain2014.dt')
    test = read_file('./IrisTest2014.dt')

    print "Total accuracy for training data: {0:.3f}".format(LDA(train, train))
    print "Total accuracy for test data: {0:.3f}".format(LDA(train, test))

def ii_1_2():
    # read in data
    train = read_file('./IrisTrain2014.dt')
    test = read_file('./IrisTest2014.dt')

    ls_tr = [float(l[0]) for l in train]  # train flower lengths
    ws_tr = [float(w[1]) for w in train]  # train flower widths

    '''
    '''
    # Find mean and variance for training data

    mu_l_tr = np.mean(ls_tr)  # mean length
    mu_w_tr = np.mean(ws_tr)  # mean width

    sigma_l_tr = np.cov(ls_tr)  # length variance
    sigma_w_tr = np.cov(ws_tr)  # width variance

    norm_train = normalize(train, mu_l_tr, sigma_l_tr, mu_w_tr, sigma_w_tr)

    check_normalization(norm_train)

    # all good, training data normalized.

    print "Total accuracy for normalized training data: {0:.3f}".format(LDA(norm_train, norm_train))

    # normalize test data as well

    norm_test = normalize(test, mu_l_tr, sigma_l_tr, mu_w_tr, sigma_w_tr)

    print "Total accuracy for normalized test data: {0:.3f}".format(LDA(norm_train, norm_test))

def ii_2_1():
    # read in files
    train = read_file('./sunspotsTrainStatML.dt')
    train = np.array(train)
    test = read_file('./sunspotsTestStatML.dt')
    test = np.array(test)

    target_vector = train[:, 5]  # extract target vector
    target_vector = target_vector.astype(np.float)
    test_target_vector = test[:, 5]
    test_target_vector = test_target_vector.astype(np.float)

    selection1 = train[:, [2, 3]]
    selection2 = train[:, 4]
    selection2 = np.array(selection2)
    selection2 = np.reshape(selection2, (len(selection2), 1))
    selection3 = train[:, [0, 1, 2, 3, 4]]

    test_selection1 = test[:, [2, 3]]
    test_selection2 = test[:, 4]
    test_selection2 = np.array(test_selection2)
    test_selection2 = np.reshape(test_selection2, (len(test_selection2), 1))
    test_selection3 = test[:, [0, 1, 2, 3, 4]]

    w1, y1 = ML_regression(selection1, target_vector)
    w2, y2 = ML_regression(selection2, target_vector)
    w3, y3 = ML_regression(selection3, target_vector)

    # plotting part

    # x and y variables of training set
    xs = [float(x) for x in selection2.flatten().tolist()]
    ys = [float(y) for y in target_vector.flatten().tolist()]

    p1, = plt.plot(xs, ys, 'b.')

    # the real variables of the test set
    xs_test_sel_2 = [float(x) for x in test_selection2.flatten().tolist()]
    ys = [float(y) for y in test_target_vector.flatten().tolist()]

    p2, = plt.plot(xs_test_sel_2, ys, 'rx')

    # predicted target variables of the test set
    ys_test_sel_2 = [] # predicted ys for selection 2
    for x in xs_test_sel_2:
        ys_test_sel_2.append(w2[0] + w2[1] * x)

    p3, = plt.plot(xs_test_sel_2, ys_test_sel_2, color = [0,1,0])

    plt.legend([p1, p2, p3], ['X and Y variables from training set', 'Real target variables from test set', 'Predicted target variables'])
    plt.savefig('figure_II_2_1_1.png')
    plt.clf()

    # error part

    # xs and ys for selection 1
    xs_test_sel_1 = test_selection1.astype(float).tolist()
    ys_test_sel_1 = []
    for xs in xs_test_sel_1:
        ys_test_sel_1.append(w1[0] + (w1[1] * xs[0]) + (w1[2]*xs[1]))

    # xs and ys for selection 3
    xs_test_sel_3 = test_selection3.astype(float).tolist()
    ys_test_sel_3 = []
    for xs in xs_test_sel_3:
        ys_test_sel_3.append(w3[0] + (w3[1] * xs[0]) + (w3[2] * xs[1]) + (w3[3] * xs[2]) + (w3[4] * xs[3]) + (w3[5] * xs[4]))


    rms_sel_1 = RMSE(ys_test_sel_1, test_target_vector)
    rms_sel_2 = RMSE(ys_test_sel_2, test_target_vector)
    rms_sel_3 = RMSE(ys_test_sel_3, test_target_vector)

    print('RMSE1: ' + str(rms_sel_1))
    print('RMSE2: ' + str(rms_sel_2))
    print('RMSE3: ' + str(rms_sel_3))

    # plot years vs prediction along with actual
    xs = xrange(1916, 2012)
    fig, ax = plt.subplots()

    p1, = ax.plot(xs, test_target_vector)
    p2, = ax.plot(xs, ys_test_sel_1)
    p3, = ax.plot(xs, ys_test_sel_2)
    p4, = ax.plot(xs, ys_test_sel_3)

    ax.set_xlim(1916, 2011)

    plt.legend([p1, p2, p3, p4], ['Actual sunspots', 'Prediction for selection 1', 'Prediction for selection 2', 'Prediction for selection 3'])
    plt.savefig('figure_II_2_1_2.png')

    plt.clf()

def ii_2_2():
    # read in files

    train = read_file('./sunspotsTrainStatML.dt')
    train = np.array(train)
    test = read_file('./sunspotsTestStatML.dt')
    test = np.array(test)

    target_vector = train[:, 5]  # extract target vector
    target_vector = target_vector.astype(np.float)
    test_target_vector = test[:, 5]
    test_target_vector = test_target_vector.astype(np.float)

    selection1 = train[:, [2, 3]]
    selection2 = train[:, 4]
    selection2 = np.array(selection2)
    selection2 = np.reshape(selection2, (len(selection2), 1))
    selection3 = train[:, [0, 1, 2, 3, 4]]

    test_selection1 = test[:, [2, 3]]
    test_selection2 = test[:, 4]
    test_selection2 = np.array(test_selection2)
    test_selection2 = np.reshape(test_selection2, (len(test_selection2), 1))
    test_selection3 = test[:, [0, 1, 2, 3, 4]]

    alphas = range(1, 100)
    err_sel1 = []
    err_sel2 = []
    err_sel3 = []

    for alpha in alphas:
        # do training
        w1 = BL_regression(alpha, selection1, target_vector)
        w2 = BL_regression(alpha, selection2, target_vector)
        w3 = BL_regression(alpha, selection3, target_vector)

        # xs and ys for selection 1
        xs_test_sel_1 = test_selection1.astype(float).tolist()
        ys_test_sel_1 = []
        for xs in xs_test_sel_1:
            ys_test_sel_1.append(w1[0] + (w1[1] * xs[0]) + (w1[2]*xs[1]))

        # xs and ys for selection 2
        xs_test_sel_2 = [float(x) for x in test_selection2.flatten().tolist()]
        ys_test_sel_2 = [] # predicted ys for selection 2
        for x in xs_test_sel_2:
            ys_test_sel_2.append(w2[0] + w2[1] * x)

        # xs and ys for selection 3
        xs_test_sel_3 = test_selection3.astype(float).tolist()
        ys_test_sel_3 = []
        for xs in xs_test_sel_3:
            ys_test_sel_3.append(w3[0] + (w3[1] * xs[0]) + (w3[2] * xs[1]) + (w3[3] * xs[2]) + (w3[4] * xs[3]) + (w3[5] * xs[4]))

        err_sel1.append(RMSE(ys_test_sel_1, test_target_vector))
        err_sel2.append(RMSE(ys_test_sel_2, test_target_vector))
        err_sel3.append(RMSE(ys_test_sel_3, test_target_vector))

    # then do the plot!

    p1, = plt.plot(alphas, err_sel1)
    plt.savefig('figure_II_2_2_1.png')
    plt.clf()
    p2, = plt.plot(alphas, err_sel2)
    plt.savefig('figure_II_2_2_2.png')
    plt.clf()
    p3, = plt.plot(alphas, err_sel3)
    plt.savefig('figure_II_2_2_3.png')
    plt.clf()
    # plt.legend([p1, p2, p3], ['Selection 1', 'Selection 2', 'Selection 3'])
    # plt.savefig('figure_II_2_2_1.png')
    plt.clf()


'''

    Helpers

'''

def BL_regression(alpha, selection, target):
    beta = 1
    (r, c) = selection.shape
    ones_vector = [1] * r
    ones_vector = np.reshape(ones_vector, (r, 1))
    phi = np.column_stack((ones_vector, selection))  # create phi_table based on the selection table
    phi = phi.astype(np.float)
    phi_T = phi.T


    sN = np.dot(phi_T, phi)
    sN *= beta # beta = 1
    (sr, sc) = sN.shape
    I = np.identity(sr)  # ones matrix
    s0 = alpha * I
    sN += s0
    sN = np.matrix(sN)
    sN = sN.I
    sN = np.array(sN)
    mN = np.dot(phi_T, target)
    mN = np.dot(mN, sN)
    mN *= beta  # wMAP solution (weights table)

    return mN


def RMSE(y, target):
    rmse = 0
    for i in range(0,len(target)):
         rmse += (target[i] - y[i])**2
    rmse /= len(target)
    rmse = math.sqrt(rmse)
    return rmse


def ML_regression(selection, target):
    (r, c) = selection.shape
    ones_vector = [1] * r
    ones_vector = np.reshape(ones_vector, (r, 1))
    phi = np.column_stack((ones_vector, selection)) # create phi_table based on the selection table
    phi = phi.astype(np.float)
    phi_t = phi.T
    phi_t_dot_phi = np.dot(phi_t, phi)
    phi_t_dot_phi_inv = np.linalg.inv(phi_t_dot_phi)
    weights_table = np.dot(phi_t_dot_phi_inv, phi_t)
    weights_table = np.dot(weights_table, target)
    weights_table = weights_table.T
    y_table = []
    for i in range(0, len(target)):
        y_table.append( np.dot(weights_table, phi[i]))
    return weights_table, y_table


def LDA(train, test):

    cs = {} # classification groups

    # split the training data into classification groups

    for t in train:
        c = int(t[2])
        try:
            cs[c].append(t)
        except KeyError:
            cs[c] = []

    '''
    '''
    # means and covariances
    means = []
    means.append(np.array(mean(cs[0])))
    means.append(np.array(mean(cs[1])))
    means.append(np.array(mean(cs[2])))

    c_1 = covariance(cs[0], means[0])
    c_2 = covariance(cs[1], means[1])
    c_3 = covariance(cs[2], means[2])

    # sum covs up and divide
    c = (1 / (len(train) - 3)) * (c_1 + c_2 + c_3)

    total_acc = 0

    for t in test:
        actual_class = int(t[2])
        mxs = []
        x = np.array([float(t[0]), float(t[1])])
        c_inv = np.linalg.inv(c)

        for i in range(0, 3):
            # calculate delta for k=i
            delta = np.dot(np.dot(x, c_inv), means[i][None].T)
            delta += -0.5 * np.dot((np.dot(means[i], c_inv)), means[i][None].T)
            delta += math.log(len(cs[0]) / len(train))

            mxs.append(delta)

        if np.argmax(mxs) == actual_class:
            total_acc += 1

    return total_acc/len(test)


# normalizes the data using
#   length_mean, length_variance,
#   width_mean and width_variance.
def normalize(pts, l_mu, l_variance, w_mu, w_variance):
    r = []
    for v in pts:
        z = []
        z.append(str((float(v[0]) - l_mu) / math.sqrt(l_variance)))  # normalized length
        z.append(str((float(v[1]) - w_mu) / math.sqrt(w_variance)))  # normalized width
        z.append(v[2])  # classification
        r.append(z)
    return r


# asserts for zero mean and unit variance
def check_normalization(pts):
    ls = [float(l[0]) for l in pts]
    ws = [float(w[1]) for w in pts]

    ls_zero_mean = np.mean(ls)
    ls_unit_var = np.cov(ls)
    ws_zero_mean = np.mean(ws)
    ws_unit_var = np.cov(ws)

    # due to floating point cut offs, the mean is very close to zero, either from
    # the left side or the right side. Therefor we check that it is within so we
    # 10^-12 error margin.
    assert ls_zero_mean < 10e-12 and ls_zero_mean > -10e-12
    assert ws_zero_mean < 10e-12 and ws_zero_mean > -10e-12
    # same goes with the variance, it is very close to one. We therefore subtract 1 from it
    # and become satisfied if the difference is less than 10^-12
    assert 1 - ls_unit_var < 10e-12
    assert 1 - ws_unit_var < 10e-12


# returns a mean vector
def mean(ds):
    v1 = [float(v[0]) for v in ds]  # feature 1
    v2 = [float(v[1]) for v in ds]  # feature 2
    return [sum(v1) / len(ds), sum(v2) / len(ds)]


def covariance(ds, mu):
    r = 0
    for x in ds:
        v = [float(x[0]), float(x[1])]
        a = np.subtract(v, mu)
        b = a[None].T
        c = np.multiply(a, b)
        # finally, sum it up
        r += c
    return r


def read_file(filename):
    data = []
    with open(filename) as rows:
        for row in csv.reader(rows, delimiter=' '):
            data.append(row)
    return data

if __name__ == '__main__':
    # ii_1_1()
    ii_1_2()
    ii_2_1()
    ii_2_2()