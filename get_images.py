import helper
from keras.preprocessing.image import ImageDataGenerator

img_dim = (128, 128)
batch_size = 16

data_dir = helper.data_dir

datagen = ImageDataGenerator(rescale=1./255)

train_path = data_dir / "train"
test_path = data_dir / "test"
val_path = data_dir / "val"

train_generator = datagen.flow_from_directory(
        train_path,
        target_size=img_dim,
        color_mode="grayscale",
        batch_size=batch_size,
        classes = ['yes', 'no'], 
        class_mode='categorical',
        shuffle=True)
test_generator = datagen.flow_from_directory(
        test_path,
        target_size=img_dim,
        color_mode="grayscale",
        batch_size=batch_size,
        classes = ['yes', 'no'],
        class_mode='categorical',
        shuffle=False)
validation_generator = datagen.flow_from_directory(
        val_path,
        target_size=img_dim,
        color_mode="grayscale",
        batch_size=batch_size,
        classes = ['yes', 'no'],
        class_mode='categorical',
        shuffle=True)

def view_images():
  import matplotlib.pyplot as plt
  for _ in range(5):
      img, label = train_generator.next()
      print(img.shape, label[0])  
      plt.imshow(img[0], cmap = "gray")
      plt.show()

if __name__ == "__main__":
  view_images()