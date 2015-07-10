import numpy as np
import cv2
import os
import os.path

HOME_DIR="/user/cruz/git/iic1103"
INPUT_DIR=HOME_DIR+"/scan-IIC1103/input-test"
TEMP_DIR= HOME_DIR+"/scan-IIC1103/temp"

#PARAMS
AREA_THRESHOLD_MIN = 1000 #5000
AREA_THRESHOLD_MAX = 10000 #7000
CORNER_WIDTH = 967
CORNER_HEIGHT = 530

#CLEAN TMP DIR
def cleanTmpDir():
	print "Cleaning tmp dir: " + TEMP_DIR
	tmpFileList = os.listdir(TEMP_DIR)
	for i in tmpFileList:
		print "Deleting " + TEMP_DIR + "/" + i
		if os.path.isfile(TEMP_DIR + "/" + i):
			os.remove(TEMP_DIR + "/" + i)

def angle_cos(p0, p1, p2):
	d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
	return abs(np.dot(d1,d2) / np.sqrt(np.dot(d1,d1)*np.dot(d2,d2)))

# square extraction idea from /usr/share/doc/opencv-doc/examples/python2/squares.py
def extractDigits(baseName,debug=False):
	fileName = INPUT_DIR+"/"+baseName+".png"
	image = cv2.imread(fileName)
	print "Read file "+ fileName
	#cut corner
	h,w,c = image.shape
	print "Original:" + str(image.shape)
	
	image_corner = image[0:CORNER_HEIGHT,w-CORNER_WIDTH:w]
	print "Cut:     " + str(image_corner.shape)
	#convert to gray
	image_gray = cv2.cvtColor(image_corner, cv2.COLOR_BGR2GRAY)
	print "Grayed:  " + str(image_gray.shape)
	cv2.imwrite(TEMP_DIR+"/"+baseName+"-corner"+".png", image_corner)
	cv2.imwrite(TEMP_DIR+"/"+baseName+"-gray"+".png", image_gray)

	image_blur = cv2.GaussianBlur(image_gray, (5,5), 0)
	image_bluc = cv2.Canny(image_blur,0,50,apertureSize=5)
	image_blcd = cv2.dilate(image_bluc, None )

	if debug:
		cv2.imshow('image_corner', image_corner)
		ch = 0xFF & cv2.waitKey()
		cv2.imshow('image_gray', image_gray)
		ch = 0xFF & cv2.waitKey()
		cv2.imshow('image_blur', image_blur)
		ch = 0xFF & cv2.waitKey()
		cv2.imshow('image_bluc', image_bluc)
		ch = 0xFF & cv2.waitKey()
		cv2.imshow('image_blcd', image_blcd)
		ch = 0xFF & cv2.waitKey()




	#retval,image_bin = cv2.threshold(image_blur,0,255,cv2.THRESH_BINARY)
	nSquare = 0
	squares = []
	rects = []
	contours, hierarchy = cv2.findContours(image_blcd, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		cnt_len = cv2.arcLength(cnt,True)
		cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
		if(len(cnt) == 4 and cv2.contourArea(cnt) > AREA_THRESHOLD_MIN and cv2.contourArea(cnt) < AREA_THRESHOLD_MAX and cv2.isContourConvex(cnt)):
			x,y,w,h = cv2.boundingRect(cnt)
			cnt = cnt.reshape(-1,2)
			max_cos = np.max([angle_cos( cnt[i], cnt[(i+1)%4], cnt[(i+2)%4]) for i in xrange(4)])
			if max_cos < 0.1:
				squareArea = cv2.contourArea(cnt)
				print "square " + str(nSquare) + ", " + str(squareArea) + ", rect: " + str(x)+","+str(y)+","+str(w)+","+str(h)+ ", area " + str(w*h)
				squares.append(cnt)
				#extract cut
				rect = image_corner[y:y+h,x:x+w]
				cv2.imwrite(TEMP_DIR+"/"+baseName+"-"+str(nSquare)+".png", rect)
				nSquare += 1

	#display squares
	print "Found " + str(len(squares)) + " squares"

	cv2.drawContours(image_corner, squares, -1, (0,255,0), 3)
	cv2.imwrite(TEMP_DIR+"/"+baseName+"-corner-squares"+".png", image_corner)

	if debug:
		cv2.imshow('squares', image_corner)
		ch = 0xFF & cv2.waitKey()
	
		# for square in squares:
		# 	image = cv2.imread(fileName)
		# 	image_corner = image[0:CORNER_HEIGHT,w-CORNER_WIDTH:w]
		# 	cv2.drawContours(image_corner, [square], -1, (0,255,0), 3)
		# 	cv2.imshow('one square', image_corner)
		# 	ch = 0xFF & cv2.waitKey()
		# 	if ch == 27:
		# 		break


		#if ch == 27:
		#	break
		cv2.destroyAllWindows()



def main():
	cleanTmpDir()

	#Get file list
	inputFileList = os.listdir(INPUT_DIR)
	for inputFile in inputFileList:
		baseName = inputFile[:inputFile.rfind('.')]
		ext = inputFile[inputFile.rfind('.')+1:]
		if ext.lower() == "png":
			print "Processing " + INPUT_DIR + "/" + inputFile
			extractDigits(baseName,debug=False)

	cv2.destroyAllWindows()



if __name__ == "__main__":
	main()
