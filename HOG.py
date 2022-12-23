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
    cellSize = (imsize//2, imsize//2) #14, 14
    blockStride = (imsize//4, imsize//4) # 7, 7
    nbins = 9
    signedGradients = True
    derivAperture = 1
    winSigma = -1.0
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64

    # define the HOG descriptor
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, 
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)

    # compute HOG descriptors
    train_descriptors = []
    for i in range(trainX.shape[0]):
        # trainX[i] = deskew(trainX[i], 28) # deskew the current image
        descriptor = hog.compute(trainX[i]) # compute the HOG features
        train_descriptors.append(descriptor) # append it to the train decriptors list

    test_descriptors = []
    for i in range(testX.shape[0]):
        # testX[i] = deskew(testX[i], 28) # deskew the current image
        descriptor = hog.compute(testX[i]) # compute the HOG features
        test_descriptors.append(descriptor) # append it to the test descriptors list

    #train_descriptors = np.array(train_descriptors)
    train_descriptors = np.resize(train_descriptors, (trainX.shape[0], 81))

    #test_descriptors = np.array(test_descriptors)
    test_descriptors = np.resize(test_descriptors, (testX.shape[0], 81))
    return train_descriptors, test_descriptors