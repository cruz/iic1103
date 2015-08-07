import sklearn
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np

import cv2
import re
import os
import sys

HOME_DIR="/user/cruz/git/iic1103"
INPUT_DIR=HOME_DIR+"/scan-IIC1103/input-test"
TEMP_DIR= HOME_DIR+"/scan-IIC1103/temp"

CLEAN_BORDER=7
MODEL_FILE="digits_cls.pkl"
AREA_THRESHOLD_MIN = 1000 #5000




def readDigit(baseName, clf, debug=True):

	img = cv2.imread(TEMP_DIR+"/"+baseName+".png")
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if debug:
		cv2.imshow(baseName+"-orig.png", img_gray)
		ch = 0xFF & cv2.waitKey()			

	img_clean = cleanImg(img_gray)
	if debug:
		cv2.imshow(baseName+"-blurred.png", img_clean)
		ch = 0xFF & cv2.waitKey()			

	img_blur = cv2.GaussianBlur(img_clean, (5, 5), 0)
	if debug:
		cv2.imshow(baseName+"-blurred.png", img_blur)
		ch = 0xFF & cv2.waitKey()			

	#Threshold the image
	ret, img_thres = cv2.threshold(img_blur, 200, 255, cv2.THRESH_BINARY_INV)
	if debug:
		cv2.imshow(baseName+"-thres.png", img_thres)
		ch = 0xFF & cv2.waitKey()			

	#Find contours
	contours, hierarchy = cv2.findContours(img_thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#Get rectangles for contours
	rects = [cv2.boundingRect(contour) for contour in contours]

	number = -1

	#Count number of big enough rectangles:
	nRects = 0
	for rect in rects:
		if rect[2]*rect[3] >= AREA_THRESHOLD_MIN:
			nRects += 1


	#For each rectangle (there should be only one), extract HOG features and predict
	print "Found " + str(nRects) + "/" + str(len(rects)) + " rectangles"
	for rect in rects:
		if rect[2]*rect[3] >= AREA_THRESHOLD_MIN:
			#Display rectangle on img
			cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,255,0), 1)
			length = int(rect[3] * 1.6) #no idea why of this
			pt1 = max(int(rect[1] + rect[3] // 2 - length // 2), 0)
			pt2 = max(int(rect[0] + rect[2] // 2 - length // 2), 0)
			#Cut roi
			roi = img_thres[pt1:pt1+length, pt2:pt2+length]
			#Display the roi in the original image
			if debug:
				cv2.imshow(baseName+".png", img)
				ch = 0xFF & cv2.waitKey()			
			#Resize the roi
			roi_scaled = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)
			roi_dilated = cv2.dilate(roi_scaled, (3,3))
			#Compute the HOG features
			roi_hog = hog(roi_dilated, orientations=9, pixels_per_cell=(14,14), cells_per_block=(1, 1), visualise=False)
			#Predict
			number = clf.predict(np.array([roi_hog], 'float64'))
			print "Predicted: " + str(number)

	cv2.imshow(baseName+".png", img)
	ch = 0xFF & cv2.waitKey()
	cv2.destroyWindow(baseName+".png")
	return number



def cleanImg(img):
	h,w = img.shape
	for i in range(0,CLEAN_BORDER):
		for j in range(0,w):
			img[i,j] = 255
	for i in range(h-CLEAN_BORDER,h):
		for j in range(0,w):
			img[i,j] = 255

	for i in range(0,h):
		for j in range(0,CLEAN_BORDER):
			img[i,j] = 255
		for j in range(w-CLEAN_BORDER,w):
			img[i,j] = 255

	return img



def train():
	print "Downloading ..."
	dataset = datasets.fetch_mldata("MNIST Original")

	features = np.array(dataset.data, 'int16')
	labels = np.array(dataset.target, 'int')

	#Compute HOGs for each image in the database
	print "Extracting features ..."
	list_hog_fd = []
	for feature in features:
		fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		list_hog_fd.append(fd)
	hog_features = np.array(list_hog_fd, 'float64')

	print "Training ..."
	clf = LinearSVC()
	clf.fit(hog_features, labels)
	joblib.dump(clf, MODEL_FILE, compress=3)
	print "Training finished ..."



def loadModel():
	print "Reading trained model ..."
	clf = joblib.load(MODEL_FILE)
	return clf



def main():

	trainMode = False
	read = True

	for arg in sys.argv:
		if arg == 'train':
			trainMode = True

	if trainMode:
		train()

	#Load trained model
	clf = loadModel()

	if clf == None:
		print "Can't find model " + str(MODEL_FILE)

	#Get file list
	inputFileList = os.listdir(TEMP_DIR)
	for inputFile in inputFileList:
		baseName = inputFile[:inputFile.rfind('.')]
		if re.match(".*-[0-9]+\.png$", inputFile) != None:
			print "Processing " + TEMP_DIR + "/" + inputFile
			digit = readDigit(baseName,clf,debug=False)
			print "Read " + str(digit)


	cv2.destroyAllWindows();

if __name__ == "__main__":
	main()
