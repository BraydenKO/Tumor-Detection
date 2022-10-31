import tensorflow as tf
import helper
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from get_images import test_generator, train_generator, validation_generator, img_dim, batch_size

path = helper.path

def get_conf_matrix(model, test_generator):
  Y_pred = model.predict(test_generator)
  y_pred = np.argmax(Y_pred, axis=1)
  y_test = test_generator.classes

  print('Confusion Matrix')
  print(confusion_matrix(y_test, y_pred))
  print('Classification Report')
  target_names = ['yes', 'no']
  print(classification_report(y_test, y_pred, target_names=target_names))

def evaluate_model(model, generator_set, keyword = ""):
  score = model.evaluate(generator_set, verbose=1)
  print(f"Model performance on {keyword}data: ")
  print(f'Loss - {keyword}data:', score[0]) 
  print(f'Accuracy - {keyword}-accuracy:', score[1])

if __name__ == "__main__":
  model = tf.keras.models.load_model(path / r"models\best model.h5")
  model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['accuracy'])
  model.summary()
  get_conf_matrix(model, test_generator)
  evaluate_model(model, test_generator, keyword = "Test-")
  evaluate_model(model, validation_generator, keyword = "Val-")
  evaluate_model(model, train_generator, keyword = "Train-")