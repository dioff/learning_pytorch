# -*- encoding: utf-8 -*-
'''
@File         :LeNet.py
@Time         :2024/01/28 14:09:32
@Author       :Lewis
@Version      :1.0
'''
import torch
from torch import nn

class Reshape(nn.Module):
	def forward(self, x):
		return x.view(-1, 1, 28, 28)


class LeNet5(nn.Module): 
	def __init__(self):
		super(LeNet5, self).__init__()
		self.net = nn.Sequential(
			Reshape(),

			# CONV1, ReLU1, POOL1
			# 输入：batch_size*1*28*28 输出：batch_size*6*28*28   (28+2*2-5+1) = 28 
			nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
			# nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
			nn.ReLU(),
			# 输出：batch_size*6*14*14
			nn.MaxPool2d(kernel_size=2, stride=2),
			
			# CONV2, ReLU2, POOL2
			# 输出：batch_size*16*10*10
			nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
			nn.ReLU(),
			# 输出：batch_size*16*5*5
			nn.MaxPool2d(kernel_size=2, stride=2),
			# 展平：batch_size*400
			nn.Flatten(),

			# FC1
			nn.Linear(in_features=16 * 5 * 5, out_features=120),
			nn.ReLU(),

			# FC2
			nn.Linear(in_features=120, out_features=84),
			nn.ReLU(),

			# FC3
			nn.Linear(in_features=84, out_features=10)
		)
		
		
	def forward(self, x):
		logits = self.net(x)
		return logits


if __name__ == '__main__':
	model = LeNet5()
	X = torch.rand(size=(256, 1, 28, 28), dtype=torch.float32)
	for layer in model.net:
		X = layer(X)
		print(layer.__class__.__name__, '\toutput shape: \t', X.shape)
	X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
	print(model(X))


