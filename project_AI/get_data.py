import get_train_categorical
import pandas as pd
import tensorflow as tf

def getXandY():
    training_dataframe = pd.read_csv('Data/New_Training.csv')

    features = training_dataframe.iloc[:, :-2]
    categs = get_train_categorical.returnCategList()

    features = tf.convert_to_tensor(features, dtype=tf.float32)
    target = tf.convert_to_tensor(categs, dtype=tf.float32)

    return (features, target)

def getNormalizedLayer(x):
    normalize = tf.keras.layers.Normalization(axis=-1)
    normalize.adapt(x)

    return normalize