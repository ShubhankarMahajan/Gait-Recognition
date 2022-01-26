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
def trainRandomForest():
    preprocess()
    global X_train1, X_test1, y_train1, y_test1
    global X_train2, X_test2, y_train2, y_test2
    global X_train3, X_test3, y_train1, y_test3
    X_train1, X_test1, y_train1, y_test1 = train_test_split(hog, features_Y, test_size=0.2, random_state=0)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(lbp, features_Y, test_size=0.2, random_state=0)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(harlick, features_Y, test_size=0.2, random_state=0)
    if os.path.exists('models/hog_rf.txt'):
        with open('models/hog_rf.txt', 'rb') as file:
            rf = pickle.load(file)
        file.close()
    else:
        rf = RandomForestClassifier()
        rf.fit(hog, features_Y)
        with open('models/hog_rf.txt', 'wb') as file:
            pickle.dump(rf, file)
        file.close()
    test(rf,"Random Forest","HOG",X_test1,y_test1,0)

    if os.path.exists('models/lbp_rf.txt'):
        with open('models/lbp_rf.txt', 'rb') as file:
            rf = pickle.load(file)
        file.close()
    else:
        rf = RandomForestClassifier()
        rf.fit(lbp, features_Y)
        with open('models/lbp_rf.txt', 'wb') as file:
            pickle.dump(rf, file)
        file.close()
    test(rf,"Random Forest","LBP",X_test2,y_test2,0)
    

    if os.path.exists('models/harlick_rf.txt'):
        with open('models/harlick_rf.txt', 'rb') as file:
            rf = pickle.load(file)
        file.close()
    else:
        rf = RandomForestClassifier()
        rf.fit(harlick, features_Y)
        with open('models/harlick_rf.txt', 'wb') as file:
            pickle.dump(rf, file)
        file.close()
    test(rf,"Random Forest","Haralick",X_test3,y_test3,30) 
os.chdir("../Comparison/")
trainRandomForest()
print("-----------------------------------------------------------------------------------------------------------------------------------------------")