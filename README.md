## 一、Niubility战队寒假视觉培训资料

![Static Badge](https://img.shields.io/badge/Pytorch-v2.0.1_-red)![Static Badge](https://img.shields.io/badge/Python-v3.10_-blue)![Static Badge](https://img.shields.io/badge/Jupyter_Notebook_-red)

> 本仓库基于Pytorch完成一些基础项目，不定期进行维护

## 二、安装依赖

[Pytorch官网地址](https://pytorch.org/)

推荐使用Anaconda安装避免安装CUDA，若安装失败请自行查找安装方式

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## 三、更新情况

- 2024.1.27 20：00 创建仓库并更新tensor操作并增加手写数字识别程序
- 2024.1.28 04：00 建立分支save更新保存模型功能
- 2024.1.28 04：50 删除save分支
- 2024.1.28 13：31 README新增icon
- 2024.1.28 13: 45 创建LeNet分支
- 2024.1.28 15: 46 增加LeNet网络，建立train和test模板，利用LeNet网络训练MNIST数据集正确率99.1%
- 2024.1.28 17：43 新增tqdm功能，实现训练过程中可视化