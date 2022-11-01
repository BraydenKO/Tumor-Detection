# Tumor-Detection
These scripts generate augmented data from a small dataset (https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
and save them if you'd like to anaylyse/see those new images.

convert_imgs.py converts all images to png (not neccessary if all images are png, but if they aren't, I suggest you run this first)

organize.py can be used to create train, test, and validation folders as well as augment and save images.  (This file is a script that should be ran)

buildCNN.py creates a model based on the VGG16 architecture 

trainCNN.py, as the name suggests, trains and saves the best model.

evalCNN.py can be used to evaluate your saved model "models\best model.h5"




