import numpy as np
import cv2
import os
import os.path

HOME_DIR="/user/cruz/git/iic1103"
INPUT_DIR=HOME_DIR+"/scan-IIC1103/input-test"
TEMP_DIR= HOME_DIR+"/scan-IIC1103/temp"

#PARAMS
AREA_THRESHOLD_MIN = 5000
AREA_THRESHOLD_MAX = 7000
CORNER_WIDTH = 1000
CORNER_HEIGHT = 650

#CLEAN TMP DIR
def cleanTmpDir():
	print "Cleaning tmp dir: " + TEMP_DIR
	tmpFileList = os.listdir(TEMP_DIR)
	for i in tmpFileList:
		print "Deleting " + TEMP_DIR + "/" + i
		if os.path.isfile(TEMP_DIR + "/" + i):
			os.remove(TEMP_DIR + "/" + i)


def extractDigits(baseName):
	fileName = INPUT_DIR+"/"+baseName+".png"
	image = cv2.imread(fileName)
	print "Read file "+ fileName
	#convert to gray
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	h,w = image_gray.shape

	print "w="+str(w) + ", " + str(w-CORNER_WIDTH)

	#image_corner = image_gray[0:CORNER_HEIGHT,w-CORNER_WIDTH:w]
	image_corner = image_gray[0:CORNER_HEIGHT,w-CORNER_WIDTH:w]

	print "Original:" + str(image.shape)
	print "Grayed:  " + str(image_gray.shape)
	print "Cutt:    " + str(image_corner.shape)

	cv2.imwrite(TEMP_DIR+"/"+baseName+"-bw"+".png", image_gray)
	cv2.imwrite(TEMP_DIR+"/"+baseName+"-cut"+".png", image_corner)

	nSquare = 0
	#from: http://stackoverflow.com/questions/11424002/how-to-detect-simple-geometric-shapes-using-opencv
	ret,image_bin = cv2.threshold(image_gray,127,255,1)
	print "Binary:  " + str(image_bin.shape)
	contours,h = cv2.findContours(image_bin,1,2)
	for cnt in contours:
		#approximate curves
		approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
		#print len(approx)
		# Take into account only squares
		if len(approx)==4:
			squareArea = cv2.contourArea(cnt)
			#print "square " + " ... " + str(approx) + ", ... " + str(approx.shape)
			#print "square " + " ... " + str(approx[0][0][0]) + ", " + str(approx[0][0][1])
			if(squareArea >= AREA_THRESHOLD_MIN and squareArea <= AREA_THRESHOLD_MAX):
				#for i in range(0,10):
				#	print str(image_gray[approx[0][0][0]+i,approx[0][0][1]:approx[0][0][1]+10])
				M = cv2.moments(cnt)
				x,y,w,h = cv2.boundingRect(approx)
				s = 0
				s0 = 0
				s1 = 0
				for i in range(x,x+w):
					for j in range(y,y+h):
						s += image_bin[i,j]
						if image_bin[i,j] == 0: s0 += 1
						else: s1 += 1
				print "square " + str(squareArea) + ", rect: " + str(x)+","+str(y)+","+str(w)+","+str(h)+ ", area " + str(w*h) + ", Sum:" + str(s) + ", 0s:" + str(s0) + ", 1s:" + str(s1)

				cv2.drawContours(image,[cnt],0,(0,0,255),-1)
				#extract cut
				square = image_gray[y:y+h,x:x+w]
				cv2.imwrite(TEMP_DIR+"/"+baseName+"-"+str(nSquare)+".png", square)
				nSquare += 1

		#TODO: store square
		#TODO: sort squares
		#TODO: process square		


	cv2.imwrite(TEMP_DIR+"/"+baseName+"-cont"+".png", image)
	cv2.imwrite(TEMP_DIR+"/"+baseName+"-bin"+".png", image_bin)






def main():
	cleanTmpDir()

	#Get file list
	inputFileList = os.listdir(INPUT_DIR)
	for inputFile in inputFileList:
		baseName = inputFile[:inputFile.rfind('.')]
		ext = inputFile[inputFile.rfind('.')+1:]
		if ext.lower() == "png":
			print "Processing " + INPUT_DIR + "/" + inputFile
			extractDigits(baseName)

if __name__ == "__main__":
	main()
