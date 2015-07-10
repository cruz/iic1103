import sklearn
import cv2
import re
import os

HOME_DIR="/user/cruz/git/iic1103"
INPUT_DIR=HOME_DIR+"/scan-IIC1103/input-test"
TEMP_DIR= HOME_DIR+"/scan-IIC1103/temp"

def readDigit(baseName, debug=True):

	img = cv2.imread(TEMP_DIR+"/"+baseName+".png")
	if debug:
		cv2.imshow(baseName+".png", img)
		ch = 0xFF & cv2.waitKey()

		
	return -1

def main():

	#Get file list
	inputFileList = os.listdir(TEMP_DIR)
	for inputFile in inputFileList:
		baseName = inputFile[:inputFile.rfind('.')]
		if re.match(".*-[0-9]+\.png$", inputFile) != None:
			print "Processing " + TEMP_DIR + "/" + inputFile
			digit = readDigit(baseName,debug=False)
			print "Read " + str(digit)


	cv2.destroyAllWindows();

if __name__ == "__main__":
	main()
