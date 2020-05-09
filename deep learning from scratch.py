import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning.exercise_7 import *
print("Setup Complete")


# 1) Start the model
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
fashion_model= Sequential()

# Add the first layer
fashion_model.add(Conv2D(12,kernel_size=3,activation='relu',input_shape=(img_rows,img_cols,1)))

#3) Add the remaining layers
fashion_model.add(Conv2D(20,kernel_size=3,activation='relu'))
fashion_model.add(Conv2D(20,kernel_size=3,activation='relu'))
fashion_model.add(Flatten())
fashion_model.add(Dense(100,activation='relu'))
fashion_model.add(Dense(num_classes, activation='softmax'))

# 4) Compile Your Model
fashion_model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

# 5) Fit The Model
fashion_model.fit(x,y,batch_size=100,epochs=4,validation_split=0.2)

# 6) Create A New Model
second_fashion_model = Sequential()
second_fashion_model.add(Conv2D(12,
                         activation='relu',
                         kernel_size=3,
                         input_shape = (img_rows, img_cols, 1)))
# Changed kernel sizes to be 2
second_fashion_model.add(Conv2D(20, activation='relu', kernel_size=2))
second_fashion_model.add(Conv2D(20, activation='relu', kernel_size=2))
# added an addition Conv2D layer
second_fashion_model.add(Conv2D(20, activation='relu', kernel_size=2))
second_fashion_model.add(Flatten())
second_fashion_model.add(Dense(100, activation='relu'))
# It is important not to change the last layer. First argument matches number of classes. Softmax guarantees we get reasonable probabilities
second_fashion_model.add(Dense(10, activation='softmax'))

second_fashion_model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])

second_fashion_model.fit(x, y, batch_size=100, epochs=4, validation_split=0.2)

#second_fashion_model.add(Conv2D(30,kernel_size=3,activation='relu',input_shape=(img_rows,img_cols,1)))
#second_fashion_model.fit(x,y,batch_size=100,epochs=4,validation_split=0.2)
