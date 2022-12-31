from sklearn.decomposition import PCA
import cv2
import joblib
from sklearn import svm
from sklearn.metrics import classification_report
import numpy as np
from dataloader import load_data
from HOG import get_HOG_feature
import matplotlib.pyplot as plt

(trainX, trainY), (testX, testY) = load_data('./dataset/')
trainX = trainX[:1000]
trainY = trainY[:1000]

train_HOG, test_HOG = get_HOG_feature(trainX, testX)
train_X=[]
train_Y=[]
for i in range(1000):
    if trainY[i] == 0:
        train_X.append(train_HOG[i])
        train_Y.append(0)
    if trainY[i] == 7:
        train_X.append(train_HOG[i])
        train_Y.append(1)
    if trainY[i] == 8:
        train_X.append(train_HOG[i])
        train_Y.append(2)


# Reduce the data to two dimensions using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(train_X)

# Train an SVM on the reduced data
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_reduced, train_Y)


# Create a mesh to plot in
h = .02  # step size in the mesh
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Obtain labels for each point in mesh. Use last trained model.
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)

# Plot also the training points
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=train_Y, cmap=plt.cm.coolwarm,edgecolor='k')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('SVM Decision Boundary')

plt.show()
