# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib.pyplot as plt
import math
import numpy
import csv


'''
    Helpers 
'''


'''

    PART I.2.X

'''

def degree_2_rad(d):
    return (d * 2 * math.pi) / 360


def gaussian(x, mean, variance):
    frac = 1 / (math.sqrt(2 * math.pi * variance ** 2))
    inner_exp = (-(1 / (2 * variance ** 2)) * (math.pow((x - mean), 2)))
    exp = math.pow(math.e, inner_exp)
    return frac * exp


# uses the following linear transformation equation to
# compute: y = mu + Lz
def multi_gaussian(z, mu, covariance):
    L = numpy.linalg.cholesky(covariance)
    z = z[None].T
    return mu + numpy.dot(L, z)


# the maximum likelihood estimate of the mean
# from CB equation 2.121 p. 112.
def mean_max_likelihood(pts):
    return sum(pts) / len(pts)


def covariance_max_likelihood(pts, mu_ml):
    r = 0
    for z in pts:
        a = numpy.subtract(z, mu_ml)
        b = a.T
        c = numpy.multiply(a, b)
        # finally, sum it up
        r += c

    return 1 / len(pts) * r


def sample_covariance_one_dimensional(pts, mu):
    return sum([(x - mu) ** 2 for x in pts]) / len(pts)


def rotate_covariance(covariance, deg):
    rad = degree_2_rad(deg)
    rot_mat = numpy.array([[numpy.cos(rad), -numpy.sin(rad)], [numpy.sin(rad), numpy.cos(rad)]])
    rot_mat_inverse = rot_mat.T
    return numpy.mat(rot_mat_inverse) * numpy.mat(covariance) * numpy.mat(rot_mat)


# the settings are
# (mean, std dev) = (−1, 1), (0, 2), and (2, 3)
def i_2_1():
    # this is always the same
    xs = [x for x in numpy.linspace(-10, 10, 200)]
    # this will depend on the setting
    ys = []

    # first setting
    for x in xs:
        ys.append(gaussian(x, -1, 1))
    line_1, = plt.plot(xs, ys)
    ys = []

    # second setting
    for x in xs:
        ys.append(gaussian(x, 0, 2))
    line_2, = plt.plot(xs, ys)
    ys = []

    # third setting
    for x in xs:
        ys.append(gaussian(x, 2, 3))
    line_3, = plt.plot(xs, ys)

    # finally, save the plots to a figure
    plt.legend([line_1, line_2, line_3], ['mu = -1, sig = 1', 'mu = 0, sig = 2', 'mu = 2, sig = 3'])
    plt.savefig('figure_I_2_1.png')
    plt.clf()


def i_2_2_and_3_and_4():
    # i_2_2 part
    covariance = numpy.array([[0.3, 0.2], [0.2, 0.2]])
    mean = numpy.array([1, 2])
    mean.shape = (2,1) # column vector!
    zs = [numpy.random.randn(2) for _ in xrange(100)]
    xs = []
    ys = []
    m_gs = []

    for z in zs:
        m_g = multi_gaussian(z, mean, covariance)
        xs.append(m_g[0])
        ys.append(m_g[1])
        m_gs.append(m_g)

    p1 = plt.scatter(xs, ys)
    plt.legend([p1], ['Sample from multivariate Gaussian distribution'])
    plt.savefig('figure_I_2_2.png')
    plt.clf()

    # i_2_3 part

    # calculate maximum likelihood mean
    mean_likelihood = (mean_max_likelihood(xs), mean_max_likelihood(ys))

    # plot it
    p1 = plt.scatter(mean_likelihood[0], mean_likelihood[1], color=[1, 0, 0])
    p2 = plt.scatter(mean[0], mean[1], color=[0, 1, 0])
    p3 = plt.scatter(xs, ys)
    plt.legend([p1, p2, p3], ['Maximum likelihood sample mean', 'Distribution mean', 'Data points'])
    plt.savefig('figure_I_2_3.png')
    plt.clf()

    # i_2_4 part
    covariance_likelihood = covariance_max_likelihood(m_gs, mean_likelihood)
    evals, evecs = numpy.linalg.eig(covariance_likelihood)

    evec1 = evecs[:, 0]
    evec1 = evec1[None].T
    evec2 = evecs[:, 1]
    evec2 = evec2[None].T

    e1 = mean + math.sqrt(evals[0]) * evec1
    e2 = mean + math.sqrt(evals[1]) * evec2

    # plot it
    # plt.quiver(mean.item(0), mean.item(1), e1, e2)
    plt.plot([e1[0], mean[0]], [e1[1], mean[1]])
    plt.plot([e2[0], mean[0]], [e2[1], mean[1]])
    p1 = plt.scatter(e1[0], e1[1], color=[1, 0, 0])
    p2 = plt.scatter(e2[0], e2[1], color=[0, 1, 0])
    p3 = plt.scatter(xs, ys)
    plt.legend([p1, p2, p3], ['Translated eigenv. one', 'Translated eigenv. two', 'Sample data'])
    plt.savefig('figure_I_2_4.png')
    plt.clf()

    # finally, rotation part

    rot_cov_ml = rotate_covariance(covariance_likelihood, 30)

    xs_rot_30 = []
    ys_rot_30 = []
    m_gs_rot_30 = []
    for z in zs:
        m_g = multi_gaussian(z, mean, rot_cov_ml)
        xs_rot_30.append(m_g[0])
        ys_rot_30.append(m_g[1])
        m_gs_rot_30.append(m_g)

    rot_cov_ml = rotate_covariance(covariance_likelihood, 60)

    xs_rot_60 = []
    ys_rot_60 = []
    m_gs_rot_60 = []
    for z in zs:
        m_g = multi_gaussian(z, mean, rot_cov_ml)
        xs_rot_60.append(m_g[0])
        ys_rot_60.append(m_g[1])
        m_gs_rot_60.append(m_g)


    rot_cov_ml = rotate_covariance(covariance_likelihood, 90)

    xs_rot_90 = []
    ys_rot_90 = []
    m_gs_rot_90 = []
    for z in zs:
        m_g = multi_gaussian(z, mean, rot_cov_ml)
        xs_rot_90.append(m_g[0])
        ys_rot_90.append(m_g[1])
        m_gs_rot_90.append(m_g)

    p1 = plt.scatter(xs, ys)
    p2 = plt.scatter(xs_rot_30, ys_rot_30, color=[1,0,0])
    p3 = plt.scatter(xs_rot_60, ys_rot_60, color=[0,1,0])
    p4 = plt.scatter(xs_rot_90, ys_rot_90, color=[0,0,1])
    plt.legend([p1, p2, p3, p4], ['Zero rotation', '30 degrees rotation', '60 degrees rotation', '90 degrees rotation'])
    plt.savefig('figure_I_2_4_rot.png')
    plt.clf()


'''

    PART I.3.X

'''


# formats an nearest neighbour tuple (distance, vector, classification)
# for output
def fmt(v):
    if v.__class__ == ().__class__:
        return '[{0:.3f}, {1:.3f}] - {2:}'.format(v[1][0], v[1][1], v[2])
    else:
        return '[{0:.3f}, {1:.3f}]'.format(v[0], v[1])


# calculates the distance from vector v
# to vector u.
def distance(v, u):
    return math.sqrt(((v[0] - u[0]) ** 2) + ((v[1] - u[1]) ** 2))


def read_file(filename):
    data = []
    with open(filename) as rows:
        for row in csv.reader(rows, delimiter = ' '):
            data.append(row)
    return data


# custom comparator that compares neighbours
# that have the format (distance, neighbour vector)
def nn_cmp(n_1, n_2):
    if n_1[0] < n_2[0]:
        return -1
    elif n_1[0] > n_2[0]:
        return 1
    else:
        return 0


# for each point, check if the class found is the same as the test
def knn(k, trains, tests, cmp):

    total_acc = 0

    for test in tests:
        acc = 0
        nns = []
        v = [float(test[0]), float(test[1])]
        c_t = test[2]

        for train in trains:
            u = [float(train[0]), float(train[1])]
            d = distance(v, u)
            c = train[2] # the classification
            nns.append((d, u, c))

        nns.sort(cmp = cmp)
        nns = nns[:k] # now its only the k nearest neighbours

        # calculate the accuracy
        for nn in nns:
            if nn[2] == c_t:
                acc += 1

        if acc / k > 0.5:
            total_acc += 1

    return total_acc / len(tests)


def nn_cross_validation(data, nr_of_folds):
    accs = {}  # map of accuracies

    ks = [x for x in xrange(1, 26, 2)]  # 1, 3, 5, ..., 25
    # split the data into 'nr_folds' chunks
    folds = [t.tolist() for t in numpy.array_split(numpy.array(data), nr_of_folds)]

    for i in range(nr_of_folds):
        # take everything except i-th chunk
        train = folds[:i] + folds[i+1:]
        # merge chunks to one list
        train = [item for chunk in train for item in chunk]
        # take i-th chunk as test data
        test = folds[i]
        # iterate over all ks
        for k in ks:
            acc = knn(k, train, test, nn_cmp)
            try:
                # map accuracy to list for each k
                accs[k].append(acc)
            except KeyError:
                accs[k] = [acc]

    for key, value in accs.iteritems():
        print 'Average accuracy for k = {0:} : {1:.3f}'.format(key, sum(value)/len(value))


def i_3_1():

    # read in the files
    data_train = read_file('./IrisTrain2014.dt')
    data_test = read_file('./IrisTest2014.dt')

    # first evaluate the training set only

    a = []
    a.append(knn(1, data_train, data_train, nn_cmp))
    a.append(knn(3, data_train, data_train, nn_cmp))
    a.append(knn(5, data_train, data_train, nn_cmp))

    # then evaluate the test-training sets
    b = []
    b.append(knn(1, data_train, data_test, nn_cmp))
    b.append(knn(3, data_train, data_test, nn_cmp))
    b.append(knn(5, data_train, data_test, nn_cmp))

    c = zip(a, b)

    for result in c:
        print '{0:.3f} (Training only) vs. {1:.3f} (With test)'.format(result[0], result[1])


def i_3_2():
    data = read_file('./IrisTrain2014.dt')
    test_data = read_file('./IrisTest2014.dt')
    print 'knn cross validation on train data'
    nn_cross_validation(data, 5)
    print
    print '3 knn accuracy on test data'
    print knn(3,data,test_data,nn_cmp)
    print


# normalizes the data using length_mean and length_variance,
# width_mean and width_variance.
def normalize(pts, l_mu, l_variance, w_mu, w_variance):
    r = []
    for v in pts:
        z = []
        z.append(str((float(v[0]) - l_mu) / math.sqrt(l_variance)))  # normalized length
        z.append(str((float(v[1]) - w_mu) / math.sqrt(w_variance)))  # normalized width
        z.append(v[2])                                               # classification
        r.append(z)
    return r


# asserts for zero mean and unit variance
def check_normalization(pts):

    ls = [float(l[0]) for l in pts]
    ws = [float(w[1]) for w in pts]

    ls_zero_mean = mean_max_likelihood(ls)
    ls_unit_var = sample_covariance_one_dimensional(ls, ls_zero_mean)
    ws_zero_mean = mean_max_likelihood(ws)
    ws_unit_var = sample_covariance_one_dimensional(ws, ws_zero_mean)

    # due to floating point cut offs, the mean is very close to zero, either from
    # the left side or the right side. Therefor we check that it is within so we
    # 10^-12 error margin.
    assert ls_zero_mean < 10e-12 and ls_zero_mean > -10e-12
    assert ws_zero_mean < 10e-12 and ws_zero_mean > -10e-12
    # same goes with the variance, it is very close to one. We therefore subtract 1 from it
    # and become satisfied if the difference is less than 10^-12
    assert 1 - ls_unit_var < 10e-12
    assert 1 - ws_unit_var < 10e-12


def i_3_3():

    train_data = read_file("./IrisTrain2014.dt")
    test_data = read_file("./IrisTest2014.dt")

    ls_tr = [float(l[0]) for l in train_data]  # train flower lengths
    ws_tr = [float(w[1]) for w in train_data]  # train flower widths

    '''
    '''
    # Find mean and variance for training data

    mu_l_tr = mean_max_likelihood(ls_tr)  # mean length
    mu_w_tr = mean_max_likelihood(ws_tr)  # mean width

    sigma_l_tr = sample_covariance_one_dimensional(ls_tr, mu_l_tr)  # length variance
    sigma_w_tr = sample_covariance_one_dimensional(ws_tr, mu_w_tr)  # width variance

    print 'Training data with mean_length: {0:.3f}, mean_width: {1:.3f}, sigma_length: {2:.3f}, sigma_width: {3:.3f}'.format(mu_l_tr, mu_w_tr, sigma_l_tr, sigma_w_tr)

    '''
    '''
    # Do normalization

    norm_tr = normalize(train_data, mu_l_tr, sigma_l_tr, mu_w_tr, sigma_w_tr)

    '''
    '''
    # do checks

    check_normalization(norm_tr)

    # '''
    # '''
    # # Perform the cross validation
    print 'knn cross validation on normalized train data'
    nn_cross_validation(norm_tr, 5)
    print  # new line

    '''
    '''
    # Now run the experiments with the test data

    # ls_te = [float(l[0]) for l in test_data]  # test flower lengths
    # ws_te = [float(w[1]) for w in test_data]  # test flower widths

    '''
    '''
    # Do normalization with factors gathered from training

    norm_te = normalize(test_data, mu_l_tr, sigma_l_tr, mu_w_tr, sigma_w_tr)

    ls_te_norm = [float(l[0]) for l in norm_te]  # test flower lengths
    ws_te_norm = [float(w[1]) for w in norm_te]  # test flower widths

    '''
    '''
    # Find mean and variance for test data and report it

    mu_l_te = mean_max_likelihood(ls_te_norm)  # mean length for test data
    mu_w_te = mean_max_likelihood(ws_te_norm)  # mean width for test data

    sigma_l_te = sample_covariance_one_dimensional(ls_te_norm, mu_l_te)  # variance length for test data
    sigma_w_te = sample_covariance_one_dimensional(ws_te_norm, mu_w_te)  # variance width for test data

    print 'Test normalized with mean_length: {0:.3f}, mean_width: {1:.3f}, sigma_length: {2:.3f}, sigma_width: {3:.3f}'.format(mu_l_te, mu_w_te, sigma_l_te, sigma_w_te)


    '''
    '''
    # training and test error of k_best = 1 (based on cross validation)
    a = knn(1, norm_tr, norm_tr, nn_cmp)
    b = knn(1, norm_tr, norm_te, nn_cmp)

    print "Training error: {0:.3f}, test error: {1:.3f}".format(1-a, 1-b)


if __name__ == '__main__':
    # i_2_1()
    # i_2_2_and_3_and_4()
    # i_3_1()
    # i_3_2()
    i_3_3()
