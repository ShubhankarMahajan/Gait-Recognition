import numpy as np
from imageio import imread
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from math import sqrt
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt
import os,cv2
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
net = Network()
net.add(FCLayer(16*16, 100))                # input_shape=(1, 16*16)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))
x = net.show_weights()
o1 = []
o2 = []

#ANCHOR/Testing
x = imread('../Testing/ml_00_3.png')
x = x.reshape(x.shape[0], 1, 16*16)
x = x.astype('float32')
x /= 255
out = net.predict(x)
for i in out:
    o1.append(sum(i[0])/10)
o1 = np.array(o1)
# #HOG
# hog_fd_anchor = get_hog_vec(imread('../Testing/ml_00_3.png'),'Testing.png')
# #LBP
# img_bgr = cv2.imread('../Testing/ml_00_3.png', 1)
# height, width, _ = img_bgr.shape
# lbp_fv = []
# img_gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
# img_lbp = np.zeros((height, width),np.uint8)
# for i in range(0, height):
#     for j in range(0, width):
#         img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
#         lbp_fv.append(img_lbp[i, j])
# o1 = np.append(o1,hog_fd_anchor)
# o1 = np.append(o1,lbp_fv)
#POSITIVE ANCHOR
files = os.listdir("../Gait Energy Image/GEI")
best = ''
best_val = 10000
for i in files:
    images = os.listdir("../Gait Energy Image/GEI/"+i)
    print("--- "+i)
    for img in images:
        print("   --- "+img+" ------------> ",end='')
        x = imread('../Gait Energy Image/GEI/'+i+'/'+img)
        x = x.reshape(x.shape[0], 1, 16*16)
        x = x.astype('float32')
        x /= 255
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
        out = net.predict(x)
        for j in out:
            o2.append(sum(j[0])/10)
        o2 = np.array(o2)
#        o2 = np.append(o2,hog_fd)
#        o2 = np.append(o2,lbp_fv)
        val = sqrt(sum( (o1 - o2)**2))
        print(val)
        if val<best_val:
            best_val = val
            best=i
        o2 = []
print("The best one seems to be with the image:"+best+"\nWith value:"+str(best_val))


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