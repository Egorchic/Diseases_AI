import pandas as pd
import numpy as np

X = pd.read_csv('Data/Training.csv')
y = pd.read_csv('Data/Testing.csv')
x = pd.DataFrame(X)

cor_matrix = x.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

x.drop(labels=to_drop, axis=1, inplace=True)
y.drop(labels=to_drop, axis=1, inplace=True)

y.to_csv('New_Testing.csv', index=False)
x.to_csv('New_Training.csv', index=False)