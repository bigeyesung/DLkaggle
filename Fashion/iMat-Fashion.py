##%%
import os
import gc
import sys
import json
import glob
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import tensorflow
import warnings
from pathlib import Path
from tqdm import tqdm
from imgaug import augmenters as iaa
from sklearn.model_selection import StratifiedKFold, KFold
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#Dowload Libraries and Pretrained Weights
# os.chdir('Mask_RCNN')
sys.path.append(str(ROOT_DIR)+'/Mask_RCNN')
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import log
import mrcnn.model as modellib

#reading files 
DATA_DIR = Path('/media/chenhsi/chenhsi/data_sets/imaterialist-fashion-2019-FGVC6')
ROOT_DIR = Path('/home/chenhsi/Projects/DLkaggle/Fashion')
IMAGE_DIR = Path('/media/chenhsi/chenhsi/data_sets/imaterialist-fashion-2019-FGVC6/train')
COCO_WEIGHTS_PATH = '/home/chenhsi/Projects/DLkaggle/Fashion/mask_rcnn_coco.h5'

# For demonstration purpose, the classification ignores attributes (only categories),
# and the image size is set to 512, which is the same as the size of submission masks
NUM_CATS = 46
IMAGE_SIZE = 512

def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
    return img

#Mask R-CNN has a load of hyperparameters. I only adjust some of them.
class FashionConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4 # a memory error occurs when IMAGES_PER_GPU is too high
    BACKBONE = 'resnet50'
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE    
    IMAGE_RESIZE_MODE = 'none'
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    #DETECTION_NMS_THRESHOLD = 0.0
    
    # STEPS_PER_EPOCH should be the number of instances 
    # divided by (GPU_COUNT*IMAGES_PER_GPU), and so should VALIDATION_STEPS;
    # however, due to the time limit, I set them so that this kernel can be run in 9 hours
#     STEPS_PER_EPOCH = 1000
#     VALIDATION_STEPS = 200
    STEPS_PER_EPOCH = 10
    VALIDATION_STEPS = 2
    
config = FashionConfig()
config.display()
class TuneDataset():
    def __init__(self):
        self.segment_df=None
        
    def SetDataset(self,file):
        #Make Datasets
        with open(DATA_DIR/"label_descriptions.json") as f:
            label_descriptions = json.load(f)
        label_names = [label['name'] for label in label_descriptions['categories']]
        attribute_names = [label['name'] for label in label_descriptions['attributes']]
        print(len(label_names))
        print(len(attribute_names))
        self.segment_df = pd.read_csv(DATA_DIR/"train.csv")
        print('le_segment_df',len(self.segment_df))
        print(self.segment_df.head())
        multilabel_percent = len(self.segment_df[self.segment_df['ClassId'].str.contains('_')])/len(self.segment_df)*100
        print(f"Segments that have attributes: {multilabel_percent:.2f}%")
        #Segments that contain attributes are only 3.46% of data, and according to the host, 80% of images have no attribute. So, in the first step, we can only deal with categories to reduce the complexity of the task

        self.segment_df['CategoryId'] = self.segment_df['ClassId'].str.split('_').str[0]
        self.segment_df['AttributeId'] = self.segment_df['ClassId'].str.split('_').str[1:]
        print("Total segments: ", len(self.segment_df))
        print('max_id:',max(list(map(lambda x:int(x),self.segment_df['CategoryId'] ))))
        self.segment_df.head()

    def show_img(self, IMG_FILE):
        I = cv2.imread(str(IMAGE_DIR) +"/"+ IMG_FILE, cv2.IMREAD_COLOR)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        I = cv2.resize(I, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
        plt.imshow(I) 
    
    def complete_make_mask(self, data, IMG_FILE):
        mask_list, cat_list = [], []
        df = data[data.ImageId == IMG_FILE].reset_index(drop = True)
        H = df.iloc[0,2]
        W = df.iloc[0,3]
        
        print("Correct Category :", sorted(set((list(df.CategoryId)))))
        # 1d mask 
        for line in df[['EncodedPixels','CategoryId']].iterrows():
            # 1d mask 
            mask = np.full(H*W,dtype='int',fill_value = -1)
            
            EncodedPixels = line[1][0]
            Category = line[1][1]
            
            pixel_loc = list(map(int,EncodedPixels.split(' ')[0::2]))
            iter_num =  list(map(int,EncodedPixels.split(' ')[1::2]))
            for p,i in zip(pixel_loc,iter_num):
                mask[p:(p+i)] = Category
            mask = mask.reshape(W,H).T
    #         print(Category, mask.shape)
            mask_list+=[mask]
            cat_list+=[Category]
        
    #     print("Output :",sorted(set(list(mask))))
    #     print('mask:\n',set(list(mask)))
    #     mask = mask.reshape(W,H).T
        #rle
    #     return mask
        return cat_list, mask_list
    def TestFunc(self):
        img_list = os.listdir(str(IMAGE_DIR))
        for k in img_list[:3]:
            cat_list1, mask_list1 = complete_make_mask(self.segment_df, k)
            plt.figure(figsize=[15,15])
            plt.subplot(3,5,1)
            show_img(k)
            plt.title('Input Image')
            i=1
            for mask, cat in zip(mask_list1, cat_list1):
                mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                plt.subplot(3,5,i+1)
                i+=1
                plt.imshow(mask,cmap='jet')
                plt.title(label_names[int(cat)])
            plt.subplots_adjust(wspace=0.4, hspace=-0.65)

        seg_att_df = self.segment_df[[len(x)>0 for x in self.segment_df['AttributeId']]].reset_index(drop=['index'])
        image_df = self.segment_df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x))
        size_df = self.segment_df.groupby('ImageId')['Height', 'Width'].mean()
        image_df = image_df.join(size_df, on='ImageId')

        # image_df = image_df.iloc[:10]
        print("Total images: ", len(image_df))
        image_df.head()
        return seg_att_df, image_df
tuner=TuneDataset()
tuner.SetDataset()
seg_att_df, image_df = tuner.TestFunc()

class FashionDataset(utils.Dataset):

    def __init__(self, df):
        super().__init__(self)
        
        # Add classes
        for i, name in enumerate(label_names):
            self.add_class("fashion", i+1, name)
        
        # Add images 
        for i, row in df.iterrows():
            self.add_image("fashion", 
                           image_id=row.name, 
                           path=str(DATA_DIR/'train'/row.name), 
                           labels=row['CategoryId'],
                           annotations=row['EncodedPixels'], 
                           height=row['Height'], width=row['Width'])

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [label_names[int(x)] for x in info['labels']]
    
    def load_image(self, image_id):
        return resize_image(self.image_info[image_id]['path'])

    def load_mask(self, image_id):
        info = self.image_info[image_id]
                
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8)
        labels = []
        
        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]
            
            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1

            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')
            sub_mask = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
            mask[:, :, m] = sub_mask
            labels.append(int(label)+1)
            
        return mask, np.array(labels)

dataset = FashionDataset(image_df)
dataset.prepare()

for i in range(8,10):
#     image_id = random.choice(dataset.image_ids)
    image_id = dataset.image_ids[i]
    print(dataset.image_reference(image_id))
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    print('mask_shape:',mask.shape)
    print('img_shape:',image.shape)
    print(class_ids)
    print(dataset.class_names)
    print(len(dataset.class_names))
#     plt.figure()
#     plt.imshow(image)
#     visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=4)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=4)


# This code partially supports k-fold training, 
# you can specify the fold to train and the total number of folds here
FOLD = 0
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
splits = kf.split(image_df) # ideally, this should be multilabel stratification

def get_fold():    
    for i, (train_index, valid_index) in enumerate(splits):
        if i == FOLD:
            return image_df.iloc[train_index], image_df.iloc[valid_index]
        
train_df, valid_df = get_fold()

train_dataset = FashionDataset(train_df)
train_dataset.prepare()
valid_dataset = FashionDataset(valid_df)
valid_dataset.prepare()

train_segments = np.concatenate(train_df['CategoryId'].values).astype(int)
print("Total train images: ", len(train_df))
print("Total train segments: ", len(train_segments))

plt.figure(figsize=(12, 3))
values, counts = np.unique(train_segments, return_counts=True)
plt.bar(values, counts)
plt.xticks(values, label_names, rotation='vertical')
plt.show()

valid_segments = np.concatenate(valid_df['CategoryId'].values).astype(int)
print("Total validation images: ", len(valid_df))
print("Total validation segments: ", len(valid_segments))

plt.figure(figsize=(12, 3))
values, counts = np.unique(valid_segments, return_counts=True)
plt.bar(values, counts)
plt.xticks(values, label_names, rotation='vertical')
plt.show()

# Note that any hyperparameters here, such as LR, may still not be optimal
LR = 1e-4
# EPOCHS = [2, 6, 8]
EPOCHS = [1, 2, 3]


warnings.filterwarnings("ignore")
model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[   'mrcnn_class_logits', 
                                                                'mrcnn_bbox_fc', 
                                                                'mrcnn_bbox', 
                                                                'mrcnn_mask'])

augmentation = iaa.Sequential([
    iaa.Fliplr(0.5) # only horizontal flip here
])
#firstly training only "head layer"
model.train(train_dataset, valid_dataset,
            learning_rate=LR*2, # train heads with higher lr to speedup learning
            epochs=EPOCHS[0],
            layers='heads',
            augmentation=None)

history = model.keras_model.history.history
model.train(train_dataset, valid_dataset,
            learning_rate=LR,
            epochs=EPOCHS[1],
            layers='all',
            augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]
#we reduce LR and train again.
model.train(train_dataset, 
            valid_dataset,
            learning_rate=LR/5,
            epochs=EPOCHS[2],
            layers='all',
            augmentation=augmentation)
new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]

#visualize training history and choose the best epoch.
epochs = range(EPOCHS[-1])
plt.figure(figsize=(18, 6))
plt.subplot(131)
plt.plot(epochs, history['loss'], label="train loss")
plt.plot(epochs, history['val_loss'], label="valid loss")
plt.legend()
plt.subplot(132)
plt.plot(epochs, history['mrcnn_class_loss'], label="train class loss")
plt.plot(epochs, history['val_mrcnn_class_loss'], label="valid class loss")
plt.legend()
plt.subplot(133)
plt.plot(epochs, history['mrcnn_mask_loss'], label="train mask loss")
plt.plot(epochs, history['val_mrcnn_mask_loss'], label="valid mask loss")
plt.legend()
plt.show()
best_epoch = np.argmin(history["val_loss"]) + 1
print("Best epoch: ", best_epoch)
print("Valid loss: ", history["val_loss"][best_epoch-1])

#prediction
glob_list = glob.glob(f'/kaggle/working/fashion*/mask_rcnn_fashion_{best_epoch:04d}.h5')
model_path = glob_list[0] if glob_list else ''

class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=ROOT_DIR)

assert model_path != '', "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
with open('MODEL.pkl', 'wb') as fid:
    pickle.dump(model, fid)

#load submission data
sample_df = pd.read_csv(DATA_DIR/"sample_submission.csv")
sample_df.head()
sample_df['EncodedPixels'][0]
# Convert data to run-length encoding
def to_rle(bits):
    rle = []
    pos = 0
    for bit, group in itertools.groupby(bits):
        group_list = list(group)
        if bit:
            rle.extend([pos, sum(group_list)])
        pos += len(group_list)
    return rle

# Since the submission system does not permit overlapped masks, 
# we have to fix them
def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois

sub_list = []
missing_count = 0
for i, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    image = resize_image(str(DATA_DIR/'test'/row['ImageId']))
    result = model.detect([image])[0]
    if result['masks'].size > 0:
        masks, _ = refine_masks(result['masks'], result['rois'])
        for m in range(masks.shape[-1]):
            mask = masks[:, :, m].ravel(order='F')
            rle = to_rle(mask)
            label = result['class_ids'][m] - 1
            sub_list.append([row['ImageId'], ' '.join(list(map(str, rle))), label])
    else:
        # The system does not allow missing ids, this is an easy way to fill them 
        sub_list.append([row['ImageId'], '1 1', 23])
        missing_count += 1

submission_df = pd.DataFrame(sub_list, columns=sample_df.columns.values)
print("Total image results: ", submission_df['ImageId'].nunique())
print("Missing Images: ", missing_count)
submission_df.head()
submission_df.to_csv("submission.csv", index=False)

#visualize the results
for i in range(9):
    image_id = sample_df.sample()['ImageId'].values[0]
    image_path = str(DATA_DIR/'test'/image_id)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = model.detect([resize_image(image_path)])
    r = result[0]
    if r['masks'].size > 0:
        masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
        for m in range(r['masks'].shape[-1]):
            masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                        (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)        
        y_scale = img.shape[0]/IMAGE_SIZE
        x_scale = img.shape[1]/IMAGE_SIZE
        rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
        
        masks, rois = refine_masks(masks, rois)
    else:
        masks, rois = r['masks'], r['rois']
        
    visualize.display_instances(img, rois, masks, r['class_ids'], 
                                ['bg']+label_names, r['scores'],
                                title=image_id, figsize=(12, 12))

  