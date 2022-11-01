# Tumor-Detection
These scripts generate augmented data from a small dataset (https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
and save them if you'd like to anaylyse/see those new images.

convert_imgs.py converts all images to png (not neccessary if all images are png, but if they aren't, I suggest you run this first)

organize.py can be used to create train, test, and validation folders as well as augment and save images.  (This file is a script that should be ran)

get_images.py is used to get the images from the train, test, and validation folder (not ran by user directly)

buildCNN.py creates a model based on the VGG16 architecture (This is called by trainCNN.py so it doesn't need to be run)

trainCNN.py, as the name suggests, trains and saves the best model.

evalCNN.py can be used to evaluate your saved model "models\best model.h5"

General order to use this project:

Run convert_imgs.py
- Run convert_imgs.py
- Run organize.py
- Run trainCNN.py
- Run evalCNN.py

Requires: 
- tensorfow
- matplotlib
- cv2
- augmentor
  
- data folder in the same folder as rest of scripts with 2 folders inside:
   - no- brains w/out tumors
   - yes- brains w/ tumors


