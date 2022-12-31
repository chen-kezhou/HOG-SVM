import cv2
from sklearn import svm
from sklearn.metrics import classification_report
# from keras.datasets import mnist
import numpy as np
from dataloader import load_data

def get_HOG_feature(trainX, testX):
    imsize = 28 # size of image (28x28)

# HOG parameters:
    winSize = (imsize, imsize) # 28, 28
    blockSize = (imsize//2, imsize//2) # 14, 14    
    cellSize = (imsize//4, imsize//4) #14, 14
    blockStride = (imsize//4, imsize//4) # 7, 7
    nbins = 9
    signedGradients = True  # 梯度是有符号的。这里与默认值相反。其他与默认值都相同。
    derivAperture = 1  # 
    winSigma = -1.0  # 不使用高斯滤波器平滑图像
    histogramNormType = 0  # 不使用归一化方法
    L2HysThreshold = 0.2  # HOG 特征规范化方式 L2-Hys 的阈值参数。此处不起作用。
    gammaCorrection = 1  # 伽马校正
    nlevels = 64  

    # define the HOG descriptor
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, 
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)

    # compute HOG descriptors
    HOG_train = []
    for i in range(trainX.shape[0]):
        descriptor = hog.compute(trainX[i]) # compute the HOG features
        HOG_train.append(descriptor) # append it to the train decriptors list

    HOG_test = []
    for i in range(testX.shape[0]):
        descriptor = hog.compute(testX[i]) # compute the HOG features
        HOG_test.append(descriptor) # append it to the test descriptors list

    HOG_train = np.reshape(HOG_train, (trainX.shape[0], -1))  # 324维的数据

    HOG_test = np.reshape(HOG_test, (testX.shape[0], -1))
    return HOG_train, HOG_test