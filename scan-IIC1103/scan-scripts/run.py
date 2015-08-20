#Found example from: https://groups.google.com/forum/#!searchin/caffe-users/mnist/caffe-users/vEvtFafhAfM/ntTyDYhkqLAJ

#import sklearn
#from sklearn.externals import joblib
#from sklearn import datasets
#from skimage.feature import hog
#from sklearn.svm import LinearSVC
import caffe

import numpy as np
import time

import cv2
import re
import os
import sys
import glob


MODEL_FILE = "/user/cruz/git/caffe/examples/mnist/lenet.prototxt"
PRETRAINED = "/user/cruz/git/caffe/examples/mnist/lenet_iter_10000.caffemodel"
net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)
TEST_IMG = "/user/cruz/git/iic1103/scan-IIC1103/opt/numbers/cut-numbers/0-3.png"
#HOME_DIR="/user/cruz/git/iic1103/"
INPUT_DIR="/user/cruz/git/iic1103/scan-IIC1103/opt/numbers/cropped_imgs"
#INPUT_DIR="/opt/numbers/cropped_imgs"
OUTPUT_DIR="/user/cruz/git/iic1103/scan-IIC1103/output"
#OUTPUT_DIR="/opt/numbers/raw_output"
#MODEL_FILE="digits_cls.pkl"
OUTPUT_FILE=OUTPUT_DIR+"/output.txt"

#CLEAN_BORDER=7
AREA_THRESHOLD_MIN = 150 

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
	#net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)
	net = caffe.Classifier(MODEL_FILE, PRETRAINED, raw_scale=255, image_dims=(28,28))
	elapsedTime = time.clock() - startTime
	print "{0:.3f}s".format(elapsedTime)
	return net

def readDigits(net,outFile,debug=False):
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
			number,conf = readSquare(square,net,debug)
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

def readSquare(square,net,debug=False):
	h,w = square.shape[0], square.shape[1]
	#print "Reading square of "+str(h) + "x" + str(w)

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
			#length = int(rect[3] * 1.6)
			length = int(rect[3] * 1.2) #extract an additional border of the image
			pt1 = max(int(rect[1] + rect[3] // 2 - length // 2), 0)
			pt2 = max(int(rect[0] + rect[2] // 2 - length // 2), 0)
			#print "RECT: " + str(rect), ", length: " + str(length) + ", pt1="+str(pt1) + ", pt2="+str(pt2)
			#print "Roi coords: Rows: " + str(pt1) + "->" + str(pt1+length) + ", Cols: " + str(pt2) + "->" + str(pt2+length)

			cv2.rectangle(square, (pt2, pt1), (pt2+length, pt1+length), (255,0,0), 1)
			#Cut roi
			roi = square_thres[pt1:pt1+length, pt2:pt2+length]
			
			#roi_padded = cv2.copyMakeBorder(roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT)
			#Resize the roi
			roi_scaled = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)
			roi_dilated = cv2.dilate(roi_scaled, (3,3))

			#print "Evaluating IMAGE: " + TEST_IMG
			#imgcaffe = caffe.io.load_image(TEST_IMG)
			#print "imgcaffe shape: " + str(imgcaffe.shape)
			#imgcaffe = imgcaffe[:,:,0]
			#print "imgcaffe shape: " + str(imgcaffe.shape)

			#imgcaffe = np.expand_dims(imgcaffe,2)

			#print "roi_dilated shape: " + str(roi_dilated.shape) + ", type: " + str(type(roi_dilated))
			imgcaffe = np.expand_dims(roi_dilated,2)
			#print "imgcaffe shape: " + str(imgcaffe.shape) + ", type: " + str(type(imgcaffe))

			prediction = net.predict([imgcaffe], oversample=False)
			number = np.argmax(prediction)
			print "Prediction: " + str(prediction) + ", ---> " + str(number)
			confidences = prediction[0]
			#out = net.forward_all(data=np.asarray(roi_dilated))
			#out = net.forward_all(data=imgcaffe)

			#Predict
			#number = clf.predict(roi_hog_np)[0]
			#confidences = clf.decision_function(roi_hog_np)[0]

			#print "Predicted: " + str(number) + ", confidence: " + str(confidences[number])
			#print str(confidences)

	if debug:
		cv2.imshow("Square", square)
		print "Square is " + str(square.shape)
		cv2.imshow("roi", roi)
		print "roi is " + str(roi.shape)
		cv2.imshow("roi_dilated", roi_dilated)
		print "roi_dilated is " + str(roi_dilated.shape)
		ch = 0xFF & cv2.waitKey()
		#cv2.destroyWindow("Square")
		#cv2.destroyWindow("roi")
		#cv2.destroyWindow("roi_dilated")
	
	if number != -1:
		return (number,confidences[number])
	else:
		return (-1,-10)

def main():

	welcome()

	#Load trained model
	net = loadModel()

	outputFile = open(OUTPUT_FILE, 'w')

	readDigits(net,outputFile,False)



if __name__ == "__main__":
	main()
