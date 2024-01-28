# -*- encoding: utf-8 -*-
'''
@File         :test.py
@Time         :2024/01/28 15:39:45
@Author       :Lewis
@Version      :1.0
'''
import torch
import cv2 as cv
from model.LeNet import LeNet5
from matplotlib import pyplot as plt

if __name__ == '__main__':
	# Loading models
	model = LeNet5()
	model.load_state_dict(torch.load('handwrittenDigitRecognition\LeNetModel.pth'))
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model.to(device)

	# READ IMAGE
	img = cv.imread('handwrittenDigitRecognition\digit.jpg')
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	gray = 255 - cv.resize(gray, (28, 28), interpolation=cv.INTER_LINEAR)
	X = torch.Tensor(gray.reshape(1, 28, 28).tolist())
	X = X.to(device)

	with torch.no_grad():
		pred = model(X)
		print(pred[0].argmax(0))
		print(pred)

