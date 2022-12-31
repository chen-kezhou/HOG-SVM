import cv2
from sklearn import svm
import joblib
from sklearn.metrics import classification_report
import numpy as np
from dataloader import load_data
from HOG import get_HOG_feature
import optparse
import logging

# 日志
logging.basicConfig(filename="HOG_SVM.log", filemode="a", 
format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO)

parser = optparse.OptionParser()
parser.add_option('-k', '--kernel', action="store", dest="kernel", help="kernel of svm",default ='poly')
options, args = parser.parse_args()
kn = options.kernel

# 加载数据集
(trainX, trainY), (testX, testY) = load_data('./dataset/')

train_HOG, test_HOG = get_HOG_feature(trainX, testX)
# print(train_HOG.shape)

# classifier
clf = svm.SVC(C=1.0, kernel = kn)
clf.fit(train_HOG, trainY)

joblib.dump(clf,f'clf_{kn}.model')
# clf = joblib.load(f'clf_{kn}.model')
# print the classification report
logging.info(f'The kernel is {kn}')
logging.info(
    f"Classification report in training set:\n"
    f"{classification_report(trainY, clf.predict(train_HOG),digits=4)}\n"
)
logging.info(
    f"Classification report in test set:\n"
    f"{classification_report(testY, clf.predict(test_HOG),digits=4)}\n"
)

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