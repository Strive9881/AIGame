
import os
import cv2
import math
import pickle
import numpy as np
from io import BytesIO
import base64
import matplotlib.pyplot as plt



'''
Function:
	图像预处理(针对截屏)
Input:
	-img(PIL.Image): 待处理的图像
	-size(tuple): 目标图像大小
	-interpolation: 插值方式
	-use_canny: 是否使用Canny算子预处理
Return:
	-img(np.array): 处理之后的图像
'''

def preprocess_img(img, size=(96, 96), interpolation=cv2.INTER_LINEAR):
	img = img[:150,:150]
	img = cv2.resize(img, size, interpolation=interpolation)
	img = cv2.Canny(img, threshold1=100, threshold2=200)
	return img


'''
Function:
	保存字典数据
Input:
	-data: dict()
	-savepath: 保存路径
	-savename: 保存文件名
'''
def save_dict(data, savepath, savename):
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	
	mode = 'wb'
	with open(os.path.join(savepath, savename), mode) as f:
		pickle.dump(data, f)


'''
Function:
	读取字典数据
Input:
	-datapath: 数据位置
'''
def read_dict(datapath):
	with open(datapath, 'rb') as f:
		return pickle.load(f)


'''
Function:
	Sigmoid函数
Input:
	-x(int): 输入数据
'''
def sigmoid(x):
	return 1.0 / (1 + math.exp(-x))