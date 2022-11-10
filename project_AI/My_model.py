import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
import get_test_categorical
import numpy as np
import os

np_config.enable_numpy_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Model(tf.Module):
    def __init__(self, normalized_layer):
        super().__init__()

        self.model = tf.keras.Sequential([
            normalized_layer,
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(41, activation='softmax')
        ])

    def compile(self):
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    def fit(self, X, Y):
        self.model.fit(X, Y, epochs=20, batch_size=64, validation_split=0.3)

    def predict(self, number_of_symtoms):
        testing_dataframe = pd.read_csv('Data/New_Testing.csv')
        x = testing_dataframe.iloc[:, :-1]
        test_categs = get_test_categorical.returnCategList()
        x_test = tf.convert_to_tensor(x, dtype=tf.float32)
        y_test = tf.convert_to_tensor(test_categs, dtype=tf.float32)

        self.model.evaluate(x_test, y_test)

        x = np.expand_dims(x_test[number_of_symtoms],
                           axis=0)  # 6 - это номер строки в 'Data/New_Testing.csv', я так проверял прогнозы.
        res = self.model.predict(x)
        index = np.argmax(res)

        predict_arr = get_test_categorical.returndiseasesList()

        return predict_arr[index]

