import matplotlib.pyplot as plt
import cv2 
import os
import numpy as np
import Augmentor
import helper
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import shutil
do_augment_data = True
do_make_ttv = True
augment_samples = 1_500

path = helper.path
rows = 2
columns = 2

data_dir = helper.data_dir

def plot_images(image_paths):
  fig = plt.figure(figsize=(15, 10))
  for idx in range(len(image_paths)):
      path = image_paths[idx]
      fig.add_subplot(rows, columns, idx+1)
      image = cv2.imread(str(path))
      plt.imshow(image)
      plt.axis('off')
  plt.show()


def traintestval(images, trainpercent, testpercent):
  indece1 = int(len(images) * trainpercent)
  indece2 = int(len(images) * (testpercent + trainpercent))
  train, test, validate = np.split(images, [indece1, indece2])
  return train, test, validate

def make_ttv_folders(train,test,val, classification):
  os.makedirs(data_dir / r"train\no", exist_ok=True)
  os.makedirs(data_dir / r"train\yes", exist_ok=True)
  
  os.makedirs(data_dir / r"test\no", exist_ok=True)
  os.makedirs(data_dir / r"test\yes", exist_ok=True)
  
  os.makedirs(data_dir / r"val\no", exist_ok=True)
  os.makedirs(data_dir / r"val\yes", exist_ok=True)
  
  for file in train:
    file_name = str(file).split("\\")[-1]
    shutil.copyfile(file, data_dir / f"train/{classification}/{file_name}")

  for file in test:
    file_name = str(file).split("\\")[-1]
    shutil.copyfile(file, data_dir / f"test/{classification}/{file_name}")
  
  for file in val:
    file_name = str(file).split("\\")[-1]
    shutil.copyfile(file, data_dir / f"val/{classification}/{file_name}")

def augment_data(images_folder, augmented_folder):
  """
  Takes images from images_folder and augments them
  to create artificial data from our limited amount of data

  Args: 
    images_folder: the path to the folder containing your "real" images
    augmented_folder: the path to the folder where you will save the artificial data
  
  Returns:
    An Augmentor Pipline object that can be used to generate data 
  """
  augmentor = Augmentor.Pipeline(images_folder, augmented_folder)
  augmentor.set_save_format("PNG")
  # Flip/Rotate
  augmentor.flip_left_right(probability=0.5)
  augmentor.flip_top_bottom(probability=0.3)
  augmentor.rotate90(probability=0.2)
  augmentor.rotate180(probability=0.2)
  augmentor.rotate270(probability=0.2)
  # Random Brightness
  augmentor.random_brightness(probability = 0.5, min_factor=0.4, max_factor=1)
  # Random Contrast
  augmentor.random_contrast(probability=0.5, min_factor=0.9, max_factor=1.5)
  # Random Distortion
  augmentor.random_distortion(probability=0.5, grid_width=3, grid_height=3, magnitude=3)
  # Zoom
  augmentor.zoom(probability=0.7, min_factor=1.1, max_factor=1.4)
  return augmentor

def change_color(image):
    image = tf.image.rgb_to_grayscale(image)
    return image

image_paths = helper.get_image_paths(data_dir)
no_images = data_dir / "no"
yes_images = data_dir / "yes"

if do_augment_data:
  no_augmentor = augment_data(no_images, data_dir / "no-augmented")
  yes_augmentor = augment_data(yes_images, data_dir / "yes-augmented")
  no_augmentor.sample(augment_samples)
  yes_augmentor.sample(augment_samples)

if do_make_ttv:
  aug_image_paths = helper.get_image_paths(data_dir, keyword="-augmented")
  image_paths[0] = image_paths[0] + aug_image_paths[0]
  image_paths[1] = image_paths[1] + aug_image_paths[1]

  # For No's
  train, test, val = traintestval(image_paths[0], 0.7, 0.1)
  make_ttv_folders(train,test,val, "no")

  # For Yes's 
  train, test, val = traintestval(image_paths[1], 0.7, 0.1)
  make_ttv_folders(train,test,val, "yes")
