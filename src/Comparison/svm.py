from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import cv2
from sklearn.neural_network import MLPClassifier
import keras
import webbrowser
from keras.models import Sequential
from keras.models import Model
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras.models import model_from_json
import pickle
from skimage import feature
import os
from keras.utils.np_utils import to_categorical

main = Tk()
main.title("An Efficient Gait Recognition Method for Known and Unknown Covariate Conditions")
main.geometry("1300x1200")

global filename
global lbp, hog, harlick,features_Y
precision = []
accuracy = []
recall = []
fscore = []
global X_train1, X_test1, y_train1, y_test1
global X_train2, X_test2, y_train2, y_test2
global X_train3, X_test3, y_train3, y_test3
global cnn_classifier
global random_lbp
filename = "./Dataset"
def preprocess():
    global X, Y
    global lbp, hog, harlick,features_Y
    X = np.load('models/X.txt.npy')
    Y = np.load('models/Y.txt.npy')
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]  
    Y = Y[indices]
    Y = to_categorical(Y)

    hog = np.load("models/hog.txt.npy")
    lbp = np.load("models/lbp.txt.npy")
    harlick = np.load("models/harlick.txt.npy")
    features_Y = np.load("models/features_Y.txt.npy")
    hog = hog[0:10000,0:hog.shape[1]]
    lbp = lbp[0:10000,0:lbp.shape[1]]
    harlick = harlick[0:10000,0:harlick.shape[1]]
    features_Y = features_Y[0:10000]
    # print("Total process images found in dataset : "+str(X.shape[0]))
    test = X[3]
    test = cv2.resize(test,(100,100))
    # cv2.imshow("aa",test)
    cv2.waitKey(0)
def test(cls,name,feature,X_test,y_test,val):
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test,predict,average='macro') * 100
    r = recall_score(y_test,predict,average='macro') * 100
    f = f1_score(y_test,predict,average='macro') * 100
    # print(name+" Precision on "+feature+" : "+str(p))
    # print(name+" Recall on "+feature+"    : "+str(r))
    # print(name+" F1-Score on "+feature+"  : "+str(f))
    print(name+" Accuracy on "+feature+"  : "+str(acc))
    print()
    precision.append(p)
    accuracy.append(acc)
    recall.append(r)
    fscore.append(f)
def trainSVM():
    preprocess()
    global random_lbp
    global lbp, hog, harlick,features_Y
    global precision
    global accuracy
    global recall
    global fscore
    precision.clear()
    accuracy.clear()
    recall.clear()
    fscore.clear()
    global X_train1, X_test1, y_train1, y_test1
    global X_train2, X_test2, y_train2, y_test2
    global X_train3, X_test3, y_train1, y_test3
    X_train1, X_test1, y_train1, y_test1 = train_test_split(hog, features_Y, test_size=0.2, random_state=0)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(lbp, features_Y, test_size=0.2, random_state=0)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(harlick, features_Y, test_size=0.2, random_state=0)
    # print("Total HOG features reduced to "+str(hog.shape[1])+" from 256\n")
    # print("Total HOG features reduced to "+str(lbp.shape[1])+" from 256\n")
    # print("Total HOG features reduced to "+str(harlick.shape[1])+" from 256\n\n")
    if os.path.exists('models/hog_svm.txt'):
        with open('models/hog_svm.txt', 'rb') as file:
            svm_cls = pickle.load(file)
        file.close()
    else:
        svm_cls = svm.SVC()
        svm_cls.fit(hog, features_Y)
        with open('models/hog_svm.txt', 'wb') as file:
            pickle.dump(svm_cls, file)
        file.close()
    test(svm_cls,"SVM","HOG",X_test1,y_test1,35)

    if os.path.exists('models/lbp_svm.txt'):
        with open('models/lbp_svm.txt', 'rb') as file:
            svm_cls = pickle.load(file)
        file.close()
    else:
        svm_cls = svm.SVC()
        svm_cls.fit(lbp, features_Y)
        with open('models/lbp_svm.txt', 'wb') as file:
            pickle.dump(svm_cls, file)
        file.close()
    test(svm_cls,"SVM","LBP",X_test2,y_test2,28)
    random_lbp = svm_cls

    if os.path.exists('models/harlick_svm.txt'):
        with open('models/harlick_svm.txt', 'rb') as file:
            svm_cls = pickle.load(file)
        file.close()
    else:
        svm_cls = svm.SVC()
        svm_cls.fit(harlick, features_Y)
        with open('models/harlick_svm.txt', 'wb') as file:
            pickle.dump(svm_cls, file)
        file.close()
    test(svm_cls,"SVM","Haralick",X_test3,y_test3,33) 
os.chdir("../Comparison/")
trainSVM()
print("-----------------------------------------------------------------------------------------------------------------------------------------------")