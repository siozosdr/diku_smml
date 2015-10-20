# to scale data (although might not be needed)
# in a train data file "traindata" and output to
# file "scaledTraindata"
./svm-scale traindata > scaledTraindata

# to calculate correct gamma and cost values based on "train data"
# produces a plot of the g and c values
python grid.py traindata

# for training based on train data file "traindata"
# if the gamma and cost we got from the privious is 0.1 and 10
./svm-train -g 0.1 -c 10 traindata

# to predict based on a test file "testdata" 
# a training model "traindata.model" 
# and the output file "prediction"
./svm-predict testdata traindata.model prediction

