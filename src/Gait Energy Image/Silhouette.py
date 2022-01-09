import os
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from cv2 import resize,imwrite
from skimage.transform import resize



print("IMPORTS DONE")
files = os.listdir('../Dataset')

for f in files:
	os.mkdir('./GEI/'+f)
	for sf in os.listdir('../Dataset/'+f):
		#REMOVE THIS CONDITION IF YOU NEED GEI FOR ALL ANGLES
		if sf.split("_")[0]!='00':
			print(sf.split("_")[1])
			break
		images = [imread('../Dataset/'+f+'/'+sf+'/'+i) for i in os.listdir('../Dataset/'+f+'/'+sf)]
		plt.imshow(images[0])
		print(sf)

		# In[4]:


		def mass_center(img,is_round=True):
			# print(img,"from mass center")
			Y = img.mean(axis=1)
			X = img.mean(axis=0)
			Y_ = np.sum(np.arange(Y.shape[0]) * Y)/np.sum(Y)
			X_ = np.sum(np.arange(X.shape[0]) * X)/np.sum(X)
			if is_round:
				return int(round(X_)),int(round(Y_))
			return X_,Y_

		def image_extract(img,newsize):
			# print(img,"from Image Extract")
			x_s = np.where(img.mean(axis=0)!=0)[0].min()
			x_e = np.where(img.mean(axis=0)!=0)[0].max()
			
			y_s = np.where(img.mean(axis=1)!=0)[0].min()
			y_e = np.where(img.mean(axis=1)!=0)[0].max()
			
			x_c,_ = mass_center(img)
		#     x_c = (x_s+x_e)//2
			x_s = x_c-newsize[1]//2
			x_e = x_c+newsize[1]//2
			img = img[y_s:y_e,x_s if x_s>0 else 0:x_e if x_e<img.shape[1] else img.shape[1]]
			return resize(img,newsize)




		images = [image_extract(i,(128,64)) for i in images]




		plt.figure()
		for i in range(10):
			plt.subplot(2,5,i+1)
			plt.imshow(images[i])
		# plt.show()




		gei = np.mean(images,axis=0)



		print("DONE WITH FILE:",f,sf);
		os.mkdir('./GEI/'+f)
		plt.imsave("./GEI/"+f+"/"+sf+".png",gei)
		# plt.show()
