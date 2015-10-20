import csv
import numpy as np
def read_file(filename):
    result = [line.rstrip('\n').split(' ') for line in open(filename)]
    return result
train = read_file('./parkinsonsTrainStatML.dt')
test = read_file('./parkinsonsTestStatML.dt')

f = open('./convertedTrain', 'w')
for t in train:
	f.write(t[-1] + ' ')
	for i in range(0,len(t)-1):
		f.write(str(i) + ':' + t[i] + ' ')
	f.write('\n')
f.close()
f = open('./convertedTest', 'w')
for t in test:
	f.write(t[-1] + ' ')
	for i in range(0,len(t)-1):
		f.write(str(i) + ':' + t[i] + ' ')
	f.write('\n')
f.close()