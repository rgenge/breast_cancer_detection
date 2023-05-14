import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Reading Data
dataset = pd.read_csv('cancer.csv')

#Cleaning data and exchanging M for 1 and B for 0
dataset = dataset.replace({'M': 1, 'B': 0})
X = dataset.iloc[:, 2:32].values
Y = dataset.iloc[:, 1].values

# Split dataset into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Scaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def models(X_train,Y_train):
  #Using Logistic Regression Algorithm to the Training Set
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state = 0,solver='lbfgs', max_iter=1000)
  log.fit(X_train, Y_train)

  #print model accuracy on the training data.
  print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
  return log
model = models(X_train,Y_train)
y_pred = model.predict(X_test)

input = input("Please enter the 30 numbers data for analisys separated by comma :\n")
input = input.split(',')
inputf= [float(i) for i in input]
inputf = [input]
input_predict = sc.transform(inputf)
prediction = model.predict(input_predict)
print("Prediction: 1 for Malignant 0 for Benign :Result= {}".format(prediction))
