import pandas as pd
import numpy as np

def returndiseasesList():
    test = pd.read_csv('Data/New_Testing.csv')
    arr = list(test.pop('prognosis').unique())

    return arr

def returnCategList():
    test = pd.read_csv('Data/New_Testing.csv')
    test_prog = test.pop('prognosis')
    test_arr = np.zeros((test.shape[0], 41)) # 41 - кол-во заболеваний
    arr = returndiseasesList()

    for i in range(test.shape[0]):
        index = arr.index(test_prog[i])
        test_arr[i][index] = 1

    return test_arr
