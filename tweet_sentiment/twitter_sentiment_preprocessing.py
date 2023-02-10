import pandas as pd

dataset = pd.read_csv('train.csv')
dataset = dataset.drop(['textID', 'text'], axis = 1)
dataset = dataset.dropna()

dataset.to_csv('train_preprocessed.csv', index = False)


