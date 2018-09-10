import cv2
import argparse
import numpy as np

# load the input image and convert it to grayscale
image = cv2.imread('IMG_20180514_165818-01.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


kernel1 = np.array([[1,0,-1],[5,0,-5],[1,0,-1]])
kernel2 = np.array([[1,5,1],[0,0,0],[-1,-5,-1]])
 

opencvOutput = cv2.filter2D(gray, -1, kernel1)
#cv2.imshow("original", gray)
cv2.imwrite("f3.jpg",opencvOutput)

opencvOutput = cv2.filter2D(gray, -1, kernel2)
#cv2.imshow("original", gray)
cv2.imwrite("f4.jpg",opencvOutput)
