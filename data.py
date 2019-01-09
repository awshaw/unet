import os
import imageio
import numpy as np 
import matplotlib.pyplot as plt
import skimage.transform as trans
from keras.preprocessing.image import ImageDataGenerator

colors = [[0,0,0], [256,256,256]]

def normalize_data(img, mask, num_class):
    if (np.max(img) > 1):
        img /= 255
        mask /= 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

def train_gen(batch_size, train_path, image_folder, mask_folder, augmentation_arguments,
              num_class = 2, target_size = (256,256), seed = None):
    image_datagen = ImageDataGenerator(**augmentation_arguments)
    mask_datagen = ImageDataGenerator(**augmentation_arguments)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = None,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = None,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = normalize_data(img,mask,num_class)
        yield (img,mask)

def test_gen(test_path,num_image = 30,target_size = (256,256)):
    for i in range(num_image):
        img = imageio.imread(os.path.join(test_path,"img_%d.png"%i),as_gray=True) / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img

def save_preds(save_path, preds, num_class = 2):
    for i,item in enumerate(preds):
        img = item[:,:,0]
        plt.imsave(os.path.join(save_path, "%d_predict.png"%i), img)