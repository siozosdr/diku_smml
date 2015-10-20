import os
import math
import numpy as np
import matplotlib.pyplot as plt
os.system ("bash -c 'echo $0'")
def read_file(filename):
    result = [line.rstrip('\n').split(',') for line in open(filename)]
    return result

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step
def sh(script):
    os.system("bash -c '%s' " % script)
    print script
'''
gamma=0.001953125
costRange = drange(0.0, 1000.0, 1)
for cost in costRange:
    sh("./libsvm-3.20//svm-train -c "+str(cost)+" -g "+str(gamma)+" data/normalizedConvertedTrain >> output.txt ")
    sh("echo -e 'Current c:' "+str(cost)+">> output.txt")
'''
costRange = [i for i in range(0, 1000)]
data = read_file('forplot.txt')
data = np.array(data)
totalnSV = data[:, 0]
totalnSV = totalnSV.tolist()
nBSV = data[:, 1]
nBSV = nBSV.tolist()
totalnSV = map(int, totalnSV)
nBSV = map(int, nBSV)
nFSV = [totalnSV[i]-nBSV[i] for i in range(len(totalnSV))]
xs = nBSV
xs2 = nFSV
p1, = plt.plot(costRange, xs, 'r')
p2, = plt.plot(costRange, xs2, 'b')
plt.xlabel('Cost')
plt.legend([p1, p2], ['No. bounded SVs', 'No. free SVs'])
plt.savefig('SVplot.png')
plt.clf()


