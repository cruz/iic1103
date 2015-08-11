import sklearn
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import datasets

import numpy as np

import time
import cv2
import sys

#TRAIN_DIR="/user/cruz/git/iic1103/scan-IIC1103/train-data"
TRAIN_DIR="./train-data"
TRAIN_MAT=TRAIN_DIR+"/mldata/mnist-original.mat"

MODEL_FILE="digits_cls.pkl"


def train(dataset):
	print "Reading dataset ..."

	features = np.array(dataset.data, 'int16')
	labels = np.array(dataset.target, 'int')
	nExamples = features.shape[0]

	#Compute HOGs for each image in the database
	print "Extracting features for " + str(nExamples) + " training examples ... ",
	sys.stdout.flush()
	startTime = time.clock()
	list_hog_fd = []
	for feature in features:
		fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		list_hog_fd.append(fd)
	hog_features = np.array(list_hog_fd, 'float64')
	elapsedTime = time.clock()-startTime
	print "{0:.3f}s ({1:.4f}s/example)".format(elapsedTime,elapsedTime/nExamples)

	print "Training ... ",
	sys.stdout.flush()
	startTime = time.clock()
	clf = LinearSVC()
	clf.fit(hog_features, labels)
	elapsedTime = time.clock()-startTime
	print "{0:.3f}s".format(elapsedTime)

	print "Saving model to " + MODEL_FILE	
	joblib.dump(clf, MODEL_FILE, compress=3)

	print "Training finished ..."




def welcome():
	print "======================================="
	print "Training script"
	print ""

def readTrainingData():
	print "Reading training data from " + TRAIN_MAT + " ... ",
	sys.stdout.flush()
	startTime = time.clock() # try with time.time(),... because time.clock() is only CPU time and time.time() is WALLTIME
	dataset = datasets.fetch_mldata("MNIST Original", data_home=TRAIN_DIR)
	elapsedTime = time.clock() - startTime
	print "{0:.3f}s".format(elapsedTime)
	return dataset

def main():

	welcome()

	dataset = readTrainingData()

	#TODO: increase training data with new examples
	#dataset = increaseTrainingData()

	train(dataset)


if __name__ == "__main__":
	main()
