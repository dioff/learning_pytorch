import cv2
import numpy as np

image = cv2.imread("Object_Count/image2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints = sift.detect(gray, None)

img_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

cv2.imwrite("Object_Count/SIFT.jpg", img_with_keypoints)
