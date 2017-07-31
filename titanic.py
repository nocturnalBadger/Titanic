import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(1)

input_data = pd.read_csv('~/Downloads/train.csv')

test_data_raw = pd.read_csv('~/Downloads/test.csv')

Y = np.array(input_data[['Survived']])

# Only pick a few columns
input_data = input_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
test_data = test_data_raw[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Fill blanks with the mode
input_data = input_data.fillna(input_data.mode().iloc[0])
test_data = test_data.fillna(test_data.mode().iloc[0])

# Numerisize gender (Yes, on a spectrum. Shut up.)
input_data['Sex'] = input_data['Sex'].map( {'female' : 1, 'male' : 0} ).astype(int)
test_data['Sex'] = test_data['Sex'].map( {'female' : 1, 'male' : 0} ).astype(int)

# Turn embarked value into a number as well
input_data['Embarked'] = input_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_data['Embarked'] = test_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# A sort of softmax, I think. Basically, get every value on a scale of 0 to 1 where the max value becomes 1.
# No clue if this is mathematically sound.
for column in input_data:
    max_recip = 1 / input_data[column].max()
    input_data[column] *= max_recip
for column in test_data:
    max_recip = 1 / test_data[column].max()
    test_data[column] *= max_recip

# Get inputs ready in np format
X = np.array(input_data)
test = np.array(test_data)


# Homemade NN stuff:

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_slope(y):
    return y * (1-y)

# Hyperparemeters
alpha = .1
hiddenSize_0 = 20
input_width = 7
iterations = 70000
dropout_rate = .05

syn0 = 2*np.random.random((input_width, hiddenSize_0)) - 1
syn1 = 2*np.random.random((hiddenSize_0, 1)) - 1

for i in range(iterations):
    l0 = X

    # Forward Propogation
    l1 = sigmoid(np.dot(l0, syn0))
    if dropout_rate != 0:  # Dropout
        l1 *= np.random.binomial([np.ones((len(X), hiddenSize_0))], 1 - dropout_rate)[0] * (1.0 / (1 - dropout_rate))
    l2 = sigmoid(np.dot(l1, syn1))

    # Backpropogation
    l2_error = l2 - Y  # l3_delta.dot(syn2.T)
    l2_delta = l2_error * sigmoid_slope(l2)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * sigmoid_slope(l1)

    syn1 -= l1.T.dot(l2_delta) * alpha
    syn0 -= l0.T.dot(l1_delta) * alpha

# Actual prediction happens here
pred_l1 = sigmoid(np.dot(test, syn0))
pred_l2 = sigmoid(np.dot(pred_l1, syn1))


# Aaaand export to csv
test_data_raw['Survived'] = prediction.astype(int)

submission = test_data_raw[['PassengerId', 'Survived']]
submission.to_csv('titanic.csv', index=False)
submission.describe()