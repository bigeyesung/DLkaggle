import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GaussianNoise, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from vit_keras import vit
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class Utility():
    def __init__(self):
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = InteractiveSession(config=self.config)

    def allocate_gpu_memory(self, gpu_number=0):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')

        if physical_devices:
            try:
                print("Found {} GPU(s)".format(len(physical_devices)))
                tf.config.set_visible_devices(physical_devices[gpu_number], 'GPU')
                tf.config.experimental.set_memory_growth(physical_devices[gpu_number], True)
                print("#{} GPU memory is allocated".format(gpu_number))
            except RuntimeError as e:
                print(e)
        else:
            print("Not enough GPU hardware devices available")

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# def allocate_gpu_memory(gpu_number=0):
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')

#     if physical_devices:
#         try:
#             print("Found {} GPU(s)".format(len(physical_devices)))
#             tf.config.set_visible_devices(physical_devices[gpu_number], 'GPU')
#             tf.config.experimental.set_memory_growth(physical_devices[gpu_number], True)
#             print("#{} GPU memory is allocated".format(gpu_number))
#         except RuntimeError as e:
#             print(e)
#     else:
#         print("Not enough GPU hardware devices available")
# allocate_gpu_memory()

# device_lib.list_local_devices()


class BirdClassfier():
    def __init__(self):
        self.train_path = "input/train"
        self.valid_path = "input/valid"
        self.test_path = "input/test"

    def DataAugment(self):
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1/255,
            horizontal_flip=True,
            rotation_range=15,
            zoom_range=0.1,
        )
        valid_datagen = ImageDataGenerator(rescale=1/255)
        test_datagen = ImageDataGenerator(rescale=1/255)

        train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=(224, 224),
            batch_size=32,
            color_mode='rgb',
            class_mode='sparse',
            shuffle=True,
        )

        validation_generator = valid_datagen.flow_from_directory(
            self.valid_path,
            target_size=(224, 224),
            batch_size=32,
            color_mode='rgb',
            class_mode='sparse')

        test_generator = test_datagen.flow_from_directory(
            self.test_path,
            target_size=(224, 224),
            batch_size=32,
            color_mode='rgb',
            class_mode='sparse')
        return train_generator, validation_generator, test_generator

    def plotImages(self,images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20,20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    
# augmented_images = [train_generator[0][0][0] for i in range(5)]
# plotImages(augmented_images)
    def Start(self, train_generator, validation_generator, test_generator):
        backend.clear_session()
        vit_model = vit.vit_l32(
            image_size=224,
            pretrained=True,
            include_top=False,
            pretrained_top=False
        )

        print(len(vit_model.layers))
        print(vit_model.layers)

        # Decay lr for each 7 epochs
        def scheduler(epoch: int, lr: float) -> float:
            if epoch != 0 and epoch % 7 == 0:
                return lr * 0.1
            else:
                return lr
        lr_scheduler_callback = LearningRateScheduler(scheduler)
        finetune_at = 28
        # fine-tuning
        for layer in vit_model.layers[:finetune_at - 1]:
            layer.trainable = False
        num_classes = len(validation_generator.class_indices)
        # Add GaussianNoise layer for robustness
        noise = GaussianNoise(0.01, input_shape=(224, 224, 3))
        # Classification head
        head = Dense(num_classes, activation="softmax")
        model = Sequential()
        model.add(noise)
        model.add(vit_model)
        model.add(head)
        model.compile(optimizer=optimizers.Adam(),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])                      
        history = model.fit(
                train_generator,
                epochs=20,
                validation_data=validation_generator,
                verbose=1, 
                shuffle=True,
                callbacks=[
                    EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
                    lr_scheduler_callback,
                ])
        history_dict = history.history
        loss_values = history_dict["loss"]
        val_loss_values = history_dict["val_loss"]
        acc_values = history_dict["accuracy"]
        val_acc_values = history_dict["val_accuracy"]
        epochs = range(1, len(history_dict["accuracy"]) + 1)

        plt.plot(epochs, loss_values, "bo", label="train")
        plt.plot(epochs, val_loss_values, "b", label="valid")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.plot(epochs, acc_values, "bo", label="train")
        plt.plot(epochs, val_acc_values, "b", label="valid")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        print("best val_acc:", np.max(val_acc_values), "epoch:", np.argmax(val_acc_values))
        print("best val_loss:", np.min(val_loss_values), "epoch:", np.argmin(val_loss_values))
        test_loss, test_acc = model.evaluate(test_generator)
        print("Test Accuracy:", test_acc)


if __name__ == "__main__":
    util = Utility()
    util.allocate_gpu_memory()
    device_lib.list_local_devices()
    classfier=BirdClassfier()
    train_generator, validation_generator, test_generator = classfier.DataAugment()
    augmented_images = [train_generator[0][0][0] for i in range(5)]
    classfier.plotImages(augmented_images)
    classfier.Start(train_generator, validation_generator, test_generator)