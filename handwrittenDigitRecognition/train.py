# -*- encoding: utf-8 -*-
'''
@File         :train.py
@Time         :2024/01/28 15:39:34
@Author       :Lewis
@Version      :1.0
'''
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.LeNet import LeNet5
from tqdm import tqdm
import matplotlib.pyplot as plt

# 定义超参数
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40

# 定义pipeline
pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

# 设置数据集
train_dataset = datasets.MNIST(
    root="handwrittenDigitRecognition\data",
    train=True,
    download=False,
    transform=pipeline
)
test_dataset = datasets.MNIST(
    root="handwrittenDigitRecognition\data",
    train=False,
    download=False,
    transform=pipeline
)
# 加载数据集
train_dataloder = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_dataloder = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# for x,y in train_dataloder:
#     print(x.shape)
#     print(y.shape)
#     break

# 加载模型并载入GPU
model = LeNet5().to(DEVICE)
# 定义优化器和损失函数
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters())
costs = []

# 定义训练方法
def train(dataloder, model, loss_func, optimizer, epoch):
    model.train()
    for batch_size, (data, target) in enumerate(tqdm(dataloder, "Train")):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        costs.append(loss)
    loss, current = loss.item(), batch_size*len(data)
    print(f'EPOCHS:{epoch + 1}\tloss:{loss:>7f}', end='\t\n')

# 定义测试方法
def test(dataloder, model, loss_func):
    model.eval() 
    test_loss, correct = 0., 0.
    size = len(dataloder.dataset)
    num_batches = len(dataloder)
    with torch.no_grad():
        for data, target in tqdm(dataloder, "Test"):
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = model(data)
            test_loss += loss_func(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: Accuracy: {(100 * correct):>0.1f}%, Average loss: {test_loss:>8f}\n')


if __name__ == "__main__":
    for epochs in range(EPOCHS):
        train(train_dataloder, model, loss_func, optimizer, epochs)
        test(test_dataloder, model, loss_func)

    # 保存模型
    torch.save(model.state_dict(), r"handwrittenDigitRecognition\result\LeNetModel_1.pth")
    print("Saved model.")

    # 绘制图片:此时的costs里面的数据是tensor类型,如果是gpu上跑的就是cuda,需要将他转换成array
    if torch.cuda.is_available():
        costs = [cost.cpu().detach().numpy() for cost in costs]
    else:
        costs = [cost.numpy() for cost in costs]
    # print(costs)
    plt.plot(costs)
    plt.xlabel("number of iteration")
    plt.ylabel("loss")
    plt.title("LeNet5")
    plt.savefig(r"handwrittenDigitRecognition\result\loss.jpg")
    plt.show()
