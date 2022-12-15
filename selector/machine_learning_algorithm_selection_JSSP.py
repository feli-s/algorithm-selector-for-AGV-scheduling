import pandas as pd
pd.set_option('display.max_columns', None)

data = pd.read_csv('instance_features_JSSP_30s.csv', sep=';')

data = data.iloc[:,1:]

#convert object to float
data["Job_AGV_ratio"]=[float(str(i).replace(",", ".")) for i in data["Job_AGV_ratio"]]
data["Skewness"]=[float(str(i).replace(",", ".")) for i in data["Skewness"]]
data["Mean operation duration per job"]=[float(str(i).replace(",", ".")) for i in data["Mean operation duration per job"]]

#delete instance feature
del data["instance"]

print(data)

#Check for missing values
#import missingno as msno
#msno.matrix(data)
#matplotlib inline
#print(msno.matrix(data))

#Transform categorical to numerical data
import sklearn
from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()
data["algorithm"] = lbe.fit_transform(data["algorithm"])

#Correlation analysis
'''import seaborn as sns
import matplotlib.pyplot as plt
corrm = data.corr()
plt.figure(figsize=(30,25))
sns.heatmap(corrm, annot=True)'''
#plt.show()

#---------------------------------------------------------------------------------------------------------------------
#Start implementing machine learning
#---------------------------------------------------------------------------------------------------------------------

#Import the train_test_split
from sklearn.model_selection import train_test_split

columns_list = list(data.columns)
X = data[columns_list[0:10]] # Input data
y = data[columns_list[-1]]

#Create the data subsets  (Train and Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) #75% train and 25% test

print("The data is ready for modelling")

#---------------------------------------------------------------------------------------------------------------------
#import evaluation metrics

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import plot_confusion_matrix # will plot the confusion matrix
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------------------------------

#DECISION TREES

from sklearn.tree import DecisionTreeClassifier

#import the metrics functions
from sklearn.metrics import (precision_score, recall_score, accuracy_score, roc_auc_score)

#create decision tree classifier object
dt_model = DecisionTreeClassifier(max_depth= 100)

#Train Decision Tree Classifier
dt_model = dt_model.fit(X_train, y_train)

#Predict the response for test dataset
y_pred_dt = dt_model.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt,average='weighted')
precision_dt = precision_score(y_test, y_pred_dt,average='weighted')
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
f1_score_dt = f1_score(y_test, y_pred_dt)

print("DT Accuracy: "+ str(accuracy_dt))
print("DT Recall: "+ str(recall_dt))
print("DT Precision: "+ str(precision_dt))
print("DT AUC Score: " +str(roc_auc_dt))
print("F1 Score: " +str(f1_score_dt))
#---------------------------------------------------------------------------------------------------------------------

#RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef

rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
matthew_corr_rf = matthews_corrcoef(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf,average='weighted')
f1_score_rf = f1_score(y_test, y_pred_rf)

print("RF Accuracy: "+ str(accuracy_rf))
print("RF Recall: "+ str(recall_rf))
print("RF Matthews Correlation: "+str(matthew_corr_rf))
print("RF Precision: "+ str(precision_rf))
print("F1 Score: " +str(f1_score_rf))

#---------------------------------------------------------------------------------------------------------------------

#Logistic Regression
from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression(max_iter=3000)

logisticRegr.fit(X_train, y_train)

y_pred_lr = logisticRegr.predict(X_test)

recall_LR = recall_score(y_test, y_pred_lr, average='weighted')
accuracy_LR = accuracy_score(y_test, y_pred_lr)
precision_LR = precision_score(y_test, y_pred_lr,average='weighted')
f1_score_LR = f1_score(y_test, y_pred_lr)


print("LR Recall: "+ str(recall_LR))
print("LR Accuracy: "+ str(accuracy_LR))
print("LR Precision: "+ str(precision_LR))
print("F1 Score: " +str(f1_score_LR))

#K NEAREST NEIGHBORS (KNN)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score

list_neigh = [18]
for n in list_neigh:
  knn_model = KNeighborsClassifier(n_neighbors=n)
  knn_model.fit(X_train, y_train)
  knn_predictions = knn_model.predict(X_test)

  knn_accuracy = accuracy_score(y_test, knn_predictions)
  knn_precision = precision_score(y_test, knn_predictions, average= 'weighted')
  recall_knn = recall_score(y_test, knn_predictions, average='weighted')
  f1_score_knn = f1_score(y_test, knn_predictions)

  print("kNN Accuracy: "+ str(n) +" "+ str(knn_accuracy))
  print("kNN Precision: "+ str(n)+" "+ str(knn_precision))
  print("kNN Recall: "+ str(recall_knn))
  print("F1 Score: " +str(f1_score_knn))


# ---------------------------------------------------------------------------------------------------------------------

#MPL (FEED FORWARD NEURAL NETWORK WITH SCIKIT LEARN
from scipy.sparse.construct import random
from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(random_state=43, hidden_layer_sizes=(50,60),
                          activation='logistic',
                          max_iter=1000).fit(X_train, y_train)

#activation {'identity', 'logistic', 'tanh', 'relu'}, default='relu'

predictions = mlp_model.predict(X_test)

mlp_accuracy = accuracy_score(y_test, predictions)
mlp_recall = recall_score(y_test, predictions, average="weighted")
mlp_precision = precision_score(y_test, predictions, average="weighted")
mlp_f1_score = f1_score(y_test, predictions, average="weighted")

print("Feed Forward Neural Network Results:")
print("MLP Accuracy: "+str(mlp_accuracy))
print("MLP Recall: "+str(mlp_recall))
print("MLP Precision: "+str(mlp_precision))
print("MLP F1-Score: "+str(mlp_f1_score))

#---------------------------------------------------------------------------------------------------------------------

#FEED FORWARD NEURAL NETWORK IMPLEMENTATION WITH KERAS
#Import libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras import metrics
import keras_metrics as km
import numpy as np
from numpy import array

#Build the feed forward neural network model
def build_nn_model():
  model = Sequential()
  model.add(Dense(100, input_dim=10, activation='relu'))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(3, activation='softmax')) #for multiclass classification
  #Compile model
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

#institate the model
nn_model = build_nn_model()

#fit the model
nn_model.fit(X_train, y_train, epochs=50, batch_size=5)

#Evaluate model
nn_model_scores = nn_model.evaluate(X_test, y_test)

#---------------------------------------------------------------------------------------------------------------------

#LSTM IMPLEMENTATION USING KERAS
#Build the feed forward neural network model
def build_lstm_model():
  model = Sequential()
  model.add(LSTM(100, return_sequences=True,input_shape=(1,len(X.columns))))
  model.add(LSTM(150, return_sequences=True))
  model.add(LSTM(210))
  model.add(Dense(3, activation='softmax')) #for multiclass classification
  #Compile the model
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])
  return model

#reshape your inputs
#Reshape the inputs. This is important because LSTM take in a 3 dimension input

#The LSTM input layer must be 3D.
#The meaning of the 3 input dimensions are: samples, time steps, and features.
#reshape input data
X_train_array = array(X_train) #array has been declared in the previous cell
print(len(X_train_array))
X_train_reshaped = X_train_array.reshape(X_train_array.shape[0],1,len(X.columns))

#reshape output data
X_test_array=  array(X_test)
X_test_reshaped = X_test_array.reshape(X_test_array.shape[0],1,len(X.columns))
#institate the model
lstm_model = build_lstm_model()

#fit the model
lstm_model.fit(X_train_reshaped, y_train, epochs=20, batch_size=5)

#Evaluate the neural network
lstm_model_scores = lstm_model.evaluate(X_test_reshaped, y_test)

#---------------------------------------------------------------------------------------------------------------------

'''def algorithm_selection_new_instance(file_analysed):
  instance_new = file_analysed.iloc[:,2:]

  #Determine algorithm for instance

  algorithm = dt_model.predict(instance_new)
  algorithm2 = rf_model.predict(instance_new)
  algorithm3 = knn_model.predict(instance_new)
  algorithm4 = mlp_model.predict(instance_new)

  algorithm = lbe.inverse_transform(algorithm)
  algorithm2 = lbe.inverse_transform(algorithm2)
  algorithm3 = lbe.inverse_transform(algorithm3)
  algorithm4 = lbe.inverse_transform(algorithm4)

  return algorithm, algorithm2, algorithm3, algorithm4


new_instance = pd.read_csv('new_instance_analysed')

algorithm, algorithm2, algorithm3, algorithm4 = algorithm_selection_new_instance(new_instance)'''


"""
After selecting the solver, the selector calculates the scheduling with this solver and prints the schedule.
"""


