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
def trainCNN():
    global X, Y
    preprocess()
    global cnn_classifier
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
    X_train, XTest, Y_train, ytest = train_test_split(X, Y, test_size=0.2, random_state=0)
    if os.path.exists('models/model.json'):
        with open('models/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn_classifier = model_from_json(loaded_model_json)
        json_file.close()    
        cnn_classifier.load_weights("models/model_weights.h5")
        cnn_classifier.make_predict_function()
    else:
        #defining CNN classifier object
        cnn_classifier = Sequential()
        #defining CNN with 32 filters to filter images 32 times and then extract important features from it
        cnn_classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1],X_train.shape[2],1), activation = 'relu'))
        #defining max pooling layer to retrieve important features from CNN
        cnn_classifier.add(MaxPooling2D(pool_size = (1, 1)))
        #another layer to filter images further
        cnn_classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
        #another max pooling layer to get important fetaures from CNN
        cnn_classifier.add(MaxPooling2D(pool_size = (1, 1)))
        #faltten will convert multi dimensional images features into single dimension
        cnn_classifier.add(Flatten())
        #defining output layer
        cnn_classifier.add(Dense(output_dim = 256, activation = 'relu'))
        #prediction layer for Y label with softmax function
        cnn_classifier.add(Dense(output_dim = Y_train.shape[1], activation = 'softmax'))
        #compile model
        cnn_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        #start training CNN with X image features and Y labels
        hist = cnn_classifier.fit(X, Y, batch_size=16, epochs=30, shuffle=True, verbose=2)
        cnn_classifier.save_weights('models/model_weights.h5')
        model_json = cnn_classifier.to_json()
        with open("models/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('models/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    # print(cnn_classifier.summary())
    predict = cnn_classifier.predict(XTest)
    predict = np.argmax(predict, axis=1)
    testLabel = np.argmax(ytest, axis=1)
    acc = accuracy_score(testLabel,predict)*100
    p = precision_score(testLabel,predict,average='macro') * 100
    r = recall_score(testLabel,predict,average='macro') * 100
    f = f1_score(testLabel,predict,average='macro') * 100
    # print("CNN Precision on Known Covariate Gait Recognition  : "+str(p))
    # print("CNN Recall on Known Covariate Gait Recognition     : "+str(r))
    # print("CNN F1-Score on Known Covariate Gait Recognition   : "+str(f))
    # print("CNN Accuracy on Known Covariate Gait Recognition   : "+str(acc))
    print("CNN Accuracy : "+str(acc))
    print()
    print("-----------------------------------------------------------------------------------------------------------------------------------------------")
    f = open('models/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']
def describe(image, eps=1e-7):
    numPoints = 24
    radius = 8
    lbp = feature.local_binary_pattern(image, numPoints,radius, method="uniform")#on image calling local binary pattern
    (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, numPoints + 3),	range=(0, numPoints + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist
os.chdir("../Comparison/")
trainCNN()