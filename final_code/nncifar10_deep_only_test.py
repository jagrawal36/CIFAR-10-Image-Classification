'''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command:
'''

from __future__ import print_function
from keras.datasets import cifar10
import numpy as np
import scipy.ndimage as si
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from IPython import embed
import pandas as pd
import h5py

batch_size = 32
nb_classes = 10
nb_epoch = 50
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

class_labels = {'airplane':0,'automobile':1,'bird':2,'cat':3,'deer':4,'dog':5,'frog':6,'horse':7,'ship':8,'truck':9}
class_inverted = dict((v, k) for k, v in class_labels.iteritems())

model = Sequential()
model.add(Convolution2D(64, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

model.load_weights("weights40epochs")
X_test1 = []
for i in range(1,300001):
  file = "../test/"+str(i)+".png"
  x = si.imread(file)
  X_test1.append(x.T)
  if i % 5000 == 0:
    print("Reading data for index = ", i)

X_test1 = np.asarray(X_test1)
X_test1 = X_test1.astype('float32')
X_test1 /= 255

print('X_test1 shape:', X_test1.shape)

y_pred = model.predict_classes(X_test1, batch_size=128, verbose=1)
#print(y_pred)
y_pred = np.asarray(y_pred)
y_class = [class_inverted[row] for row in y_pred]
out = pd.DataFrame(y_class)
out.index += 1
out.to_csv('out1.csv')
embed()
