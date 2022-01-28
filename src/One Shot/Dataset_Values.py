import sys
import numpy as np
from keras.layers import *
from imageio import imread
from math import sqrt
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from keras.preprocessing import image
import  numpy  as  np
from keras.layers import *
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import model_from_json
import os
os.chdir("../One Shot")
def set_val():
    with open('models/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
    json_file.close()    
    model.load_weights("models/model_weights.h5")
    model.make_predict_function()
    acc = 1
    final_predictions = []
    files = os.listdir("../Gait Energy Image/GEI")
    best = ''
    best_val = 10000
    fh = open("Dataset_Values.txt","w")
    for i in files:
        images = os.listdir("../Gait Energy Image/GEI/"+i)
        # print("--- "+i)
        for img in images:
            fname = img
            training_img = '../Gait Energy Image/GEI/'+i+'/'+img
            img = image.load_img(training_img, target_size=(240, 320))
            training_img_data = image.img_to_array(img)
            training_img_data = np.expand_dims(training_img_data, axis=0)
            training_img_data = preprocess_input(training_img_data)
            vgg_feature_2 = model.predict(training_img_data)
            vgg_feature_2= np.array(vgg_feature_2[0])
            #HOG
    #        hog_fd = get_hog_vec(x,i+'_'+img)
            #LBP
    #        img_bgr = cv2.imread('../Gait Energy Image/GEI/'+i+'/'+img, 1)
    #        height, width, _ = img_bgr.shape
    #        lbp_fv = []
    #        img_gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    #        img_lbp = np.zeros((height, width),np.uint8)
    #        for a in range(0, height):
    #            for b in range(0, width):
    #                img_lbp[a, b] = lbp_calculated_pixel(img_gray, a, b)
    #                lbp_fv.append(img_lbp[a, b])
    #        o2 = np.append(o2,hog_fd)
    #        o2 = np.append(o2,lbp_fv)
            vgg_feature_2 = str(list(vgg_feature_2))
            fh.write(i+"-"+fname+"<--->"+vgg_feature_2+"\n")
def display_val():
    f = open("Dataset_Values.txt","r")
    for i in f.readlines():
        x,y = i.split("<--->")
        y = np.array(eval(y))
set_val()