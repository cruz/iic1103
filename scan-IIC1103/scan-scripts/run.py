import sklearn
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
import time

import cv2
import re
import os
import sys
import glob

#HOME_DIR="/user/cruz/git/iic1103/"
#INPUT_DIR="/user/cruz/git/iic1103/scan-IIC1103/opt/numbers"
INPUT_DIR="/opt/numbers/cropped_imgs"
#OUTPUT_DIR="/user/cruz/git/iic1103/scan-IIC1103/output"
OUTPUT_DIR="/opt/numbers/raw_output"
MODEL_FILE="digits_cls.pkl"
OUTPUT_FILE=OUTPUT_DIR+"/output.txt"

#CLEAN_BORDER=7
AREA_THRESHOLD_MIN = 200 

INPUT_IMGS_DIR=INPUT_DIR+"/test"

# [x, y, w, h]
SQUARE_COORDINATES = [ [ 68, 39, 42, 42],
					   [116, 39, 42, 42],
					   [164, 39, 42, 42],
					   [212, 39, 42, 42],
					   [260, 39, 42, 42],
					   [308, 39, 42, 42],
					   [356, 39, 42, 42],
					   [404, 39, 42, 42],
					   [494, 39, 42, 42], 

					   [ 68,137, 42, 42],
					   [116,137, 42, 42],

					   [212,137, 42, 42],

					   [308,137, 42, 42],
					   [356,137, 42, 42],

					   [448,137, 42, 42],
					   [494,137, 42, 42], 
					 ]


def welcome():
	print "======================================="
	print "Digits recognizer script"
	print ""

def loadModel():
	print "Reading trained model ...",
	sys.stdout.flush()
	startTime = time.clock()
	clf = joblib.load(MODEL_FILE)
	elapsedTime = time.clock() - startTime
	print "{0:.3f}s".format(elapsedTime)
	return clf

def readDigits(clf,outFile,debug=False):
	#Get list of images
	print "Reading input images from " + INPUT_IMGS_DIR,
	sys.stdout.flush()
	imgFileList = glob.glob(INPUT_IMGS_DIR+"/*.png")
	print "... found " + str(len(imgFileList)) + " PNG's"

	for imgFile in imgFileList:
		baseName = imgFile[imgFile.rfind('/')+1:imgFile.rfind('.')]
		print "Reading image " + baseName
		img = cv2.imread(imgFile)

		if debug:
			cv2.imshow(baseName+".png", img)
			#ch = 0xFF & cv2.waitKey()
			#cv2.destroyWindow(baseName+".png")

		#Extract squares from images
		squares = []
		numbers = []
		confidences = []

		for coords in SQUARE_COORDINATES:
			x,y,w,h = coords
			square = img[y:y+h,x:x+w]
			squares.append(square)

		if debug and False:
			i=0
			for square in squares:
				cv2.imshow("Square " + str(i),square)
				ch = 0xFF & cv2.waitKey()
				cv2.destroyWindow("Square "+ str(i))
				i+=1

		for square in squares:
			number,conf = readSquare(square,clf,False)
			numbers.append(number)
			confidences.append(conf)

		line = baseName + ".png" + " " + writeNum(numbers[0]) + writeNum(numbers[1]) + writeNum(numbers[2]) \
									   + writeNum(numbers[3]) + writeNum(numbers[4]) + writeNum(numbers[5]) \
									   + writeNum(numbers[6]) + writeNum(numbers[7]) \
									   + "-" + writeNum(numbers[8])
		line += " " + writeNum(numbers[9]) + writeNum(numbers[10]) 
		line += " " + writeNum(numbers[11])
		line += " " + writeNum(numbers[12]) + writeNum(numbers[13]) 
		line += " " + writeNum(numbers[14]) + writeNum(numbers[15]) 

		line += " {0:.2f} {1:.2f} {2:.2f} {3:.2f} {4:.2f} {5:.2f} {6:.2f} {7:.2f} {8:.2f}".format(confidences[0], \
					confidences[1], confidences[2], confidences[3], confidences[4], confidences[5], confidences[6],
					confidences[7], confidences[8])

		line += " {0:.2f} {1:.2f} {2:.2f} {3:.2f} {4:.2f} {5:.2f} {6:.2f}".format(confidences[9], \
					confidences[10], confidences[11], confidences[12], confidences[13], confidences[14], confidences[15])

		print line
		outFile.write(line+"\n")

		if debug:
			ch = 0xFF & cv2.waitKey()
			cv2.destroyWindow(baseName+".png")

def writeNum(n):
	if n == -1:
		return "X"
	return str(n)

def readSquare(square,clf,debug=False):
	h,w = square.shape[0], square.shape[1]
	print "Reading square of "+str(h) + "x" + str(w)

	square_gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
	square_blur = cv2.GaussianBlur(square_gray, (5, 5), 0)
	ret, square_thres = cv2.threshold(square_blur, 200, 255, cv2.THRESH_BINARY_INV)

	#Find contours
	contours, hierarchy = cv2.findContours(square_thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#Get rectangles for contours
	rects = [cv2.boundingRect(contour) for contour in contours]

	number = -1
	#Count number of big enough rectangles:
	nRects = 0
	for rect in rects:
		#print "Trying rect " + str(rect) + ", area = " + str(rect[2]*rect[3])
		if rect[2]*rect[3] >= AREA_THRESHOLD_MIN:
			nRects += 1


	#For each rectangle (there should be only one), extract HOG features and predict
	#print "Found " + str(nRects) + "/" + str(len(rects)) + " rectangles"
	if nRects < 1:
		print "WARNING. Square coulnd't find a rectangle big enough"
		for rect in rects:
			print "Rect " + str(rect) + ", area = " + str(rect[2]*rect[3])
		if debug:
			cv2.imshow("Square", square)
			ch = 0xFF & cv2.waitKey()
			cv2.destroyWindow("Square")			

	for rect in rects:
		if rect[2]*rect[3] >= AREA_THRESHOLD_MIN:
			#Display rectangle on img
			cv2.rectangle(square, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,255,0), 1)
			length = int(rect[3] * 1.6) #no idea why of this
			pt1 = max(int(rect[1] + rect[3] // 2 - length // 2), 0)
			pt2 = max(int(rect[0] + rect[2] // 2 - length // 2), 0)
			#Cut roi
			roi = square_thres[pt1:pt1+length, pt2:pt2+length]
			#Resize the roi
			roi_scaled = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)
			roi_dilated = cv2.dilate(roi_scaled, (3,3))
			#Compute the HOG features
			roi_hog = hog(roi_dilated, orientations=9, pixels_per_cell=(14,14), cells_per_block=(1, 1), visualise=False)
			roi_hog_np = np.array([roi_hog], 'float64')
			#Predict
			number = clf.predict(roi_hog_np)[0]
			confidences = clf.decision_function(roi_hog_np)[0]

			print "Predicted: " + str(number) + ", confidence: " + str(confidences[number])
			print str(confidences)

	if debug:
		cv2.imshow("Square", square)
		ch = 0xFF & cv2.waitKey()
		cv2.destroyWindow("Square")
	
	if number != -1:
		return (number,confidences[number])
	else:
		return (-1,-10)

def main():

	welcome()

	#Load trained model
	clf = loadModel()

	outputFile = open(OUTPUT_FILE, 'w')

	readDigits(clf,outputFile,False)



if __name__ == "__main__":
	main()
