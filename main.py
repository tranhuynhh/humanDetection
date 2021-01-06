import os
import cv2
import torch
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def getHOG() : 
    winSize = (128,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog

def createDataTarget(hog):
    data = []
    target = []
    folders = ['./Data/Positive', './Data/Negative']
    for folder in folders:
        imgs = os.listdir(folder)
        for img in imgs:
            this_img = cv2.imread(folder+'/'+img)
            this_img = cv2.resize(this_img, (128, 64), interpolation=cv2.INTER_CUBIC)
            this_img = cv2.cvtColor(this_img, cv2.COLOR_BGR2RGB)
            descriptor = hog.compute(this_img)
            data.append(np.array(descriptor).ravel())
            target.append(1 if folder=='./Data/Positive' else -1)
    return data, target

def main():
    hog = getHOG()
    data, target = createDataTarget(hog)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True, random_state=40)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(prf(y_test, y_pred))

main()