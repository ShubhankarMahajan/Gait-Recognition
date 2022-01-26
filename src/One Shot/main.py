import numpy as np
from keras.layers import *
from imageio import imread
from math import sqrt
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from keras.preprocessing import image
# from scipy.misc import imsave
import  numpy  as  np
from keras.layers import *
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dropout, Flatten, Dense
# from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
import os,cv2,pickle
#DEF
#For HOG
def get_hog_vec(img,name):
    resized_img = resize(img, (128*4, 64*4))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
    # plt.imsave(name, hog_image, cmap="gray")
    return fd
#FOR LBP
def get_pixel(img, center, x, y):
	new_value = 0
	try:
		if img[x][y] >= center:
			new_value = 1
	except:
		pass
	return new_value
def lbp_calculated_pixel(img, x, y):
	center = img[x][y]
	val_ar = []
	val_ar.append(get_pixel(img, center, x-1, y-1))
	val_ar.append(get_pixel(img, center, x-1, y))
	val_ar.append(get_pixel(img, center, x-1, y + 1))
	val_ar.append(get_pixel(img, center, x, y + 1))
	val_ar.append(get_pixel(img, center, x + 1, y + 1))
	val_ar.append(get_pixel(img, center, x + 1, y))
	val_ar.append(get_pixel(img, center, x + 1, y-1))
	val_ar.append(get_pixel(img, center, x, y-1))
	power_val = [1, 2, 4, 8, 16, 32, 64, 128]
	val = 0
	for i in range(len(val_ar)):
		val += val_ar[i] * power_val[i]
	return val

#---------------------------------<><><><><><><>---------------------------------
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(240, 320, 3), padding='VALID'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='VALID'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Block 2
model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
model.add(AveragePooling2D(pool_size=(19, 19)))

# set of FC => RELU layers
os.chdir("../One Shot")
# print(os.getcwd())
model.add(Flatten())
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.save_weights('./models/model_weights.h5')
model_json = model.to_json()
with open("./models/model.json", "w") as json_file:
    json_file.write(model_json)
json_file.close()    
f = open('./models/history.pckl', 'wb')
pickle.dump(model.history, f)
f.close()
acc = 1
final_predictions = []
f = open("Accuracy.txt","w")
for testing_image_file in os.listdir("../Testing"):
    testing_img = '../Testing/'+testing_image_file
    img = image.load_img(testing_img, target_size=(240, 320))
    testing_img_data = image.img_to_array(img)
    testing_img_data = np.expand_dims(testing_img_data, axis=0)
    testing_img_data = preprocess_input(testing_img_data)

    vgg_feature_1 = model.predict(testing_img_data)
    vgg_feature_1= np.array(vgg_feature_1[0])
    files = os.listdir("../Gait Energy Image/GEI")
    best = ''
    best_val = 10000
    for i in files:
        images = os.listdir("../Gait Energy Image/GEI/"+i)
        # print("--- "+i)
        for img in images:
            # print("   --- "+img+" ------------> ",end='')
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
            val = sqrt(sum( (vgg_feature_1 - vgg_feature_2)**2))
            # print(val)
            if val<best_val:
                best_val = val
                best=i
    if testing_image_file.split(".")[0].split("_")[0]==best:
        acc+=1
    final_predictions.append((testing_image_file,best,best_val))
    # print("Tested with:"+testing_image_file.split(".")[0].split("_")[0]+"\tPredicted:"+best+"\tDistance:"+str(best_val))
accuracy_results = '''Tested With\t\t\tPredicted\t\t\tDistance\n'''
for i in final_predictions:
    accuracy_results += i[0].split(".")[0].split("_")[0]+"\t\t\t"+i[1]+"\t\t\t"+str(i[2])+"\n"
accuracy_results += "\nAccuracy:"+str((acc/17)*100)+"%"
print(accuracy_results)
f.write(accuracy_results)
f.close()
# ------------------------------------------<><><><><><><><><>------------------------------------------
'''
#NEGATIVE ANCHOR
x = imread('../Gait Energy Image/GEI/fyc/00_1.png')
x = x.reshape(x.shape[0], 1, 16*16)
x = x.astype('float32')
x /= 255
out = net.predict(x)
print("\n")
print("predicted values : ")
print(len(out))
for i in out:
    o2.append(sum(i[0])/10)
# print(o1)
# print(o2)
print("NEGATIVE ANCHOR")
o1 = np.array(o1)
o2 = np.array(o2)
print(sqrt(sum( (o1 - o2)**2)))
'''