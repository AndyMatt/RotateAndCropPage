#!/usr/bin/python3

import sys
import os
import cv2
import numpy as np

def ProcessFile(input, output):
	##Read file as input
	img = cv2.imread(input)

	##Blur source image to remove artifacts
	blurred = cv2.blur(img, (20,20))

	##Desaturate Source to make it easier to find contours
	imgray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
	
	##Change the threshold of the levels to produce simple geometry
	th, threshed = cv2.threshold(imgray, 120, 255, 0)

	##Find Contours in Source
	cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

	##Create a copy of the threshold result
	canvas  = threshed.copy()

	## sort and choose the largest contour
	cnts = sorted(cnts, key = cv2.contourArea)
	cnt = cnts[-2]

	## approx the contour, so the get the corner points
	arclen = cv2.arcLength(cnt, True)
	approx = cv2.approxPolyDP(cnt, 0.02* arclen, True)
	cv2.drawContours(canvas, [cnt], -1, (255,0,0), 5, cv2.LINE_AA)
	cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 5, cv2.LINE_AA)
	
	##Calculate a bounding box
	contours_op, hierarchy_op = cv2.findContours(threshed, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	cnts = sorted(cnts, key = cv2.contourArea)
	cnt = cnts[-1]

	##Calculate angle of bounding box for rotation
	_, _, angle = rect = cv2.minAreaRect(cnt)
	if(angle > 45): angle -= 90
	(h,w) = img.shape[:2]
	(center) = (h//2,h//2)

	##Process the Rotation
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(img, M, (int(w),int(h)), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

	##Process rotate image
	imgray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

	##Change the Level Threshold Again
	th, threshed = cv2.threshold(imgray, 150, 200, 0)
	cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
	
	##Find the Contours of the bounding box
	cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	# Determine the Bounding box using the contours
	for c in cnts:
		x,y,w,h = cv2.boundingRect(c)
		ROI = rotated[y-10:y+h+20, x-10:x+w+20]
		break

	##Export Result
	cv2.imwrite(output,ROI)

if(len(sys.argv) < 3):
	print("Usage: Rotate.py Filename Output")
	exit(0)
if(sys.argv[1] == "-l") :
	for i in sys.argv[2:]:
		output = os.path.splitext(i)[0] + "c.png"
		ProcessFile(i,output)
else:
	output = os.path.splitext(sys.argv[1])[0] + "c.png"
	ProcessFile(sys.argv[1], output)
