"""

This Codes from https://github.com/PacktPublishing/OpenCV-3-x-with-Python-By-Example/tree/master/Chapter01


"""
import cv2

img = cv2.imread('captain.png')
cv2.imshow('Image', img)

cv2.waitKey()