import cv2
import numpy as np
img = cv2.imread('Object_Count/image2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

draw_img = img.copy()
res1 = cv2.drawContours(draw_img, contours, 0, (0, 0, 255), 2)

cv2.imwrite("thresh.jpg" ,thresh)
cv2.imwrite("coutours_drawing.jpg", res1)
cv2.waitKey(0)