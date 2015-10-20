# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: disha
"""

from matplotlib import pyplot as plt
import math
import numpy as np

def generate_gaussian(mu, sigma):
    return lambda t: 1.0/(math.sqrt(2*math.pi*sigma**2))*np.exp(-1.0/(2*sigma**2)*(t-mu)**2)

#{def multivariate_gaussian_distribution():

    #x = np.linspace(-10,10, num=1000)
    #g1 = generate_gaussian(-1,1)
    #g2 = generate_gaussian(0,2)
    #g3 = generate_gaussian(2,3)

    #pylab.plot(x,g1(x))
    #pylab.plot(x,g2(x))
    #pylab.plot(x,g3(x))
    #pylab.show()

def multivariate_gaussian_distribution(points=100, do_three_too=False, do_four_too=True):

    mu = [1,2]
    covariance = [[0.3, 0.2],[0.2,0.2]]
    L = np.linalg.cholesky(covariance)

    mv_gaussians = [mu + np.dot(L,np.random.randn(2)) for _ in xrange(points)]

    xs = [mv_gaussians[i][0] for i in xrange(points)]
    ys = [mv_gaussians[i][1] for i in xrange(points)]

    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)
    ax.scatter(xs,ys)

    ax.set_xlim([-1, 4])
    ax.set_ylim([0, 4])
    ax.set_title('multivariate gaussian')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()
    
    mean = (sum(xs)/len(xs), sum(ys)/len(xs))

    if do_three_too:
       
       ax.scatter(*mean,color=[1,0,0])
       ax.scatter(mu[0],mu[1],color=[0,1,0])
       print math.sqrt((mean[0]-mu[0])**2+(mean[1]-mu[1])**2)
       
    if do_four_too:

        covariance = [np.subtract(x, mean) for x in mv_gaussians]



multivariate_gaussian_distribution(100, do_four_too=True)