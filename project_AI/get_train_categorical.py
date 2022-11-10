import pandas as pd
import numpy as np

def returndiseasesList():
    test = pd.read_csv('Data/New_Testing.csv')
    arr = list(test.pop('prognosis').unique())

    return arr

def returnCategList():
    train = pd.read_csv('Data/New_Training.csv')
    train_prog = train.pop('prognosis')
    train_arr = np.zeros((train.shape[0], 41)) # 41 - кол-во заболеваний
    arr = returndiseasesList()

    for i in range(train.shape[0]):
        index = arr.index(train_prog[i])
        train_arr[i][index] = 1

    return train_arr




