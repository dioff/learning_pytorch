import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import random

folder = glob.glob("first/*.jpg") 

def plot_resolution():
    img_size_list = []
    
    for img in folder:
        img_i = Image.open(img)
        img_i_size = img_i.size
        img_size_list.append(img_i_size)

    # print(img_size_list)

    width_list = [img_size_list[i][0] for i in range(len(img_size_list))]
    hight_list = [img_size_list[i][1] for i in range(len(img_size_list))]

    width_list_bias = []
    height_list_bias = []
    # 添加bias本数据集较多相同大小图片
    for i in width_list:
        bias = random.uniform(0.1, 0.9)
        i_bias = i + bias
        width_list_bias.append(i_bias)

    for j in hight_list:
        bias = random.uniform(0.1, 0.9)
        j_bias = j + bias
        height_list_bias.append(j_bias)

    print(len(set(width_list_bias)))
    print(len(set(height_list_bias)))
    # plt.rcParams['font.sans--serif'] = ['SimHei']
    # plt.rcParams['font.size'] = 8
    plt.rcParams['axes.unicode_minus'] = False

    plt.scatter(width_list_bias, height_list_bias, s=1)
    plt.xlabel('width') 
    plt.ylabel('height') 
    plt.title("height-width")
    plt.show()

    

plot_resolution()
    
