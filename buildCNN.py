import tensorflow as tf
from get_images import img_dim
# Using VGG16 Architecture
# https://arxiv.org/pdf/1409.1556v6.pdf

# Create a model
model = tf.keras.Sequential()
# First Layer - Conv (1)
model.add(tf.keras.layers.Conv2D(input_shape=img_dim + (1,),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# Second Layer - Conv (2)
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# Third Layer - Max Pooling
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
# Fourth - Fifth Layer - Conv (3)
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# Fifth Layer - Conv (4)
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# Sixth Layer - Max Pooling
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
# Seventh Layer - Conv (5)
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# Eight Layer - Conv (6)
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# Ninth Layer - Conv (7)
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# Tenth Layer - Max Pooling
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
# Eleventh Layer - Conv (8)
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# Twelfth Layer - Conv (9)
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# Thirteenth Layer - Conv (10)
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# Fourteeth Layer - Max Pooling
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
# Fifteeth Layer - Conv (11)
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# Sixteenth Layer - Conv (12)
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# Seventeenth Layer - Conv (13)
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# Eighteenth Layer - Max Pooling
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
# Flatten
model.add(tf.keras.layers.Flatten())
# Nineteenth Layer - Dense (14)
model.add(tf.keras.layers.Dense(units=4096,activation="relu"))
# Twentieth Layer - Dense (15)
model.add(tf.keras.layers.Dense(units=4096,activation="relu"))
# Twenty-first Layer - Dense (16)
model.add(tf.keras.layers.Dense(units=2, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['accuracy'])

if __name__ == "__main__":
  model.summary()
  #tf.keras.utils.plot_model(model, to_file = helper.path, show_shapes=True)