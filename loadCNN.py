import tensorflow as tf
import helper
import numpy as np

img_dim = (128, 128)
batch_size = 16

path = helper.path

def classify_images(model, images_path):
  image_files = list(images_path.glob("*"))

  images = []
  for file in image_files:
    img = tf.keras.utils.load_img(str(file), target_size=img_dim, color_mode='grayscale' )
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images.append(x)

  images = np.vstack(images)

  Y_pred = model.predict(images)
  y_pred = np.argmax(Y_pred, axis=1)

  for idx, prediction in enumerate(y_pred):
    if prediction == 0:
      print(f'Model predicted yes with {Y_pred[idx][0]} confidence ', Y_pred[idx])
    else:
      print(f'Model predicted no with {Y_pred[idx][1]} confidence ', Y_pred[idx])
    print(f'Image: {image_files[idx]}')

if __name__ == "__main__":
  model = tf.keras.models.load_model(path / r"models\best model.h5")
  classify_images(model, path / r"data\train\yes")