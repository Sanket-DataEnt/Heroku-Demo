# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#reading the dataset
dataset = pd.read_csv('hiring.csv')

# print(dataset.shape)
# imputing missing values of experience as a 0 i.e they are freshers.
dataset['experience'].fillna(0, inplace=True)

# filling missing values of test_score with mean test score
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

# Preparing Independent Variable
X = dataset.iloc[:, :3]

# Creating function to convert String to integer
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))


# Selecting Target Variable
y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))


