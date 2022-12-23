import cv2
from sklearn import svm
from sklearn.metrics import classification_report
import numpy as np
from dataloader import load_data
from HOG import get_HOG_feature

# 加载数据集
(trainX, trainY), (testX, testY) = load_data('./dataset/')

# 
trainX = trainX[:5000] 
trainY = trainY[:5000]
testX = testX[:5000]
testY = testY[:5000]

train_HOG, test_HOG = get_HOG_feature(trainX, testX)

# classifier
clf = svm.SVC(C=1.0, kernel='rbf')
clf.fit(train_HOG, trainY)

# print the classification report
print(classification_report(testY, clf.predict(test_HOG)))

# visualize the predictions

'''
for i in range(testX.shape[0]):
    # resize the image to be 10x bigger
    img = cv2.resize(testX[i], None, fx=10, fy=10)
    # make prediction on the current image
    prediction = clf.predict(test_descriptors[i:i+1])
    # write the predicted number on the image
    cv2.putText(img, 'prediction:' + str(prediction[0]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255)
    # display the image
    cv2.imshow('img', img)
    
    # get the pressed key
    key = cv2.waitKey(0)
    # if the pressed key is q, destroy the window and break out of the loop
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
'''