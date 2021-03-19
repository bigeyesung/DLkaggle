import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import itertools


#GPU Support for TensorFlow
ret=tf.test.is_gpu_available()
gpus=tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
#import data
base_dir = '/media/chenhsi/chenhsi/data_sets2/Intel'

# Getting training and testing directories
train_dir = os.path.join(base_dir, 'seg_train', 'seg_train')
test_dir = os.path.join(base_dir, 'seg_test','seg_test')
pred_dir = os.path.join(base_dir, 'seg_pred','seg_pred')

# Directory with the different training pictures
train_buildings = os.path.join(train_dir, 'buildings')
train_forest = os.path.join(train_dir, 'forest')
train_glacier = os.path.join(train_dir, 'glacier')
train_mountain = os.path.join(train_dir, 'mountain')
train_sea = os.path.join(train_dir, 'sea')
train_street = os.path.join(train_dir, 'street')

# Directory with the different testing pictures
test_buildings = os.path.join(test_dir, 'buildings')
test_forest = os.path.join(test_dir, 'forest')
test_glacier = os.path.join(test_dir, 'glacier')
test_mountain = os.path.join(test_dir, 'mountain')
test_sea = os.path.join(test_dir, 'sea')
test_street = os.path.join(test_dir, 'street')

#Looking into the names of the different files in train_buildings and train_forest. 
#The files are simply given numbers.
train_building_fnames = os.listdir(train_buildings)
train_forest_fnames = os.listdir(train_forest)
print(train_building_fnames[:10])
print(train_forest_fnames[:10])

#see how many training and testing images I have in each category.
training_images_len = []
for category in os.listdir(train_dir):
    num_images = len(os.listdir(os.path.join(train_dir, category)))
    training_images_len.append(num_images)
    print(f'total training {category} images:', num_images)

print(f'All training images: {np.sum(training_images_len)}')
print('-'*50)

testing_images_len = []
for category in os.listdir(test_dir):
    num_images = len(os.listdir(os.path.join(test_dir, category)))
    testing_images_len.append(num_images)
    print(f'total testing {category} images:', num_images)
print(f'All testing images: {np.sum(testing_images_len)}')


#This shows 4 pictures for each category. Each row shows a different category.
nrows = 6
ncols = 4
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)
# get the images to show from each category. I am going to show 4 images per category, one category for each row
images_to_show = []
for category in os.listdir(train_dir):
    images_to_show.append([os.path.join(train_dir, category, fname) 
                           for fname in os.listdir(os.path.join(train_dir, category))[0:ncols]])
# The previous code outputs a list of lists, this flattens the list
images_to_show = list(itertools.chain.from_iterable(images_to_show))
for i, img_path in enumerate(images_to_show):
    # set up subplot (indices start at 1)
    sp = plt.subplot(nrows, ncols, i+1)
    sp.axis('Off') # Don't show axes or gridlines
    
    img = mpimg.imread(img_path)
    plt.imshow(img)
# plt.show()

#Load the data
#Set labels to be inferred so that they are generated from the directory structure. 
#Set image_size to (150,150) since that is the size of the images
# train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir, 
#                                                                     labels='inferred',  
#                                                                     batch_size=32, 
#                                                                     image_size=(150,150))
# type(train_dataset)
# classes = train_dataset.class_names
# classes

#Create ImageDataGenerator
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                                    rescale = 1.0/255,
                                                    shear_range = 0.2,
                                                    zoom_range = 0.2
)
train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(150,150), 
                                                    class_mode="categorical", 
                                                    shuffle=True, seed = 5)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255)
validation_generator = validation_datagen.flow_from_directory(  test_dir, 
                                                                target_size=(150,150), 
                                                                class_mode='categorical',
                                                                shuffle=True, seed = 5)

#creating models
num_classes = 6
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), 
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # dropout layer
    tf.keras.layers.Dropout(0.2),
    # 256 neuron layer
    tf.keras.layers.Dense(256, activation='relu'),
    # dropout layer
    tf.keras.layers.Dropout(0.2),
    # Only 6 output neurons since there are 6 categories
    tf.keras.layers.Dense(6, activation= 'softmax')  
])
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics = ['accuracy'])

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.80):
            print("\nReached 80% accuracy so cancelling training")
            self.model.stop_training = True
callbacks = myCallback()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                              min_delta=0, 
                                              patience=3, 
                                              verbose=0,
                                              mode='auto', 
                                              baseline=None, 
                                              restore_best_weights=False
                                             )
# initially, I was going to use model.fit_generator, 
# but this is now deprecated as of tensorflow-gpu==2.2.0 or higher
history = model.fit(train_generator, 
                    batch_size = 20, 
                    epochs=5, 
                    verbose = 1, 
                    validation_data = validation_generator)
model.fit_generator

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()