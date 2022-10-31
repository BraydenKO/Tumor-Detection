import tensorflow as tf
import os
import helper
from buildCNN import model
from get_images import train_generator, validation_generator, test_generator, batch_size

EPOCHS = 100
num_samples = len(train_generator.classes)
path = helper.path

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

model.summary()
os.makedirs("models", exist_ok=True)

# Create ModelCheckpoint object
checkpointer = tf.keras.callbacks.ModelCheckpoint(monitor='val_accuracy', mode = 'max',
                               filepath=path / r"models\best model2.h5", 
                               verbose=1, save_best_only=True)
# Create EarlyStopping object
early_stopping_monitor = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience = 20)
callbacks_list = [checkpointer, early_stopping_monitor]

hist = model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator, steps_per_epoch=num_samples//batch_size, callbacks = callbacks_list)

score = model.evaluate(test_generator, verbose=1)
print("Model performance on test data: ")
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

import matplotlib.pyplot as plt
print("Accuracy Plot: ")
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()