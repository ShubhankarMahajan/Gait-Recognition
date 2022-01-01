import numpy as np
from imageio import imread
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime
from math import sqrt
from keras.datasets import mnist
from keras.utils import np_utils
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
net = Network()
net.add(FCLayer(16*16, 100))                # input_shape=(1, 16*16)    ;   output_shape=(1, 100)
# net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
# net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
# net.add(ActivationLayer(tanh, tanh_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
# test on 3 samples
o1 = []
o2 = []
x = imread('../Gait Energy Image/GEI/fyc/00_1.png')
x = x.reshape(x.shape[0], 1, 16*16)
x = x.astype('float32')
x /= 255
out = net.predict(x)
print("\n")
print("predicted values : ")
print(len(out))
for i in out:
    o1.append(sum(i[0])/10)
x = imread('../Gait Energy Image/GEI/fyc/00_3.png')
x = x.reshape(x.shape[0], 1, 16*16)
x = x.astype('float32')
x /= 255
out = net.predict(x)
print("\n")
print("predicted values : ")
print(len(out))
for i in out:
    o2.append(sum(i[0])/10)
print("POSITIVE ANCHOR")
# print(o1)
# print(o2)
o1 = np.array(o1)
o2 = np.array(o2)
print(sqrt(sum( (o1 - o2)**2)))
o2 = []
x = imread('../Gait Energy Image/GEI/ljg/00_1.png')
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