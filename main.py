import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

label_encoder = LabelEncoder()
standard = StandardScaler()
import pandas as pd

data = pd.read_csv("Churn_Modelling.csv", index_col=[0])
# print(data)
X = data.iloc[:, 3:12]
Y = data.iloc[:, -1]
rated_dummies = pd.get_dummies(data.Geography)
final_data = pd.concat([X, rated_dummies], axis=1)

# print(final_data.columns)
final_data["Gender"] = label_encoder.fit_transform(final_data["Gender"])
final_data.drop(columns=["Geography"], axis=1, inplace=True)
print(final_data.columns)

X_train, X_test, Y_train, Y_test = train_test_split(final_data, Y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

sequential = Sequential()
sequential.add(Dense(units=10, kernel_initializer="he_uniform", activation="relu", input_dim=11))
sequential.add(Dense(units=20, kernel_initializer="he_uniform", activation="relu", input_dim=10))
sequential.add(Dense(units=1, kernel_initializer="glorot_uniform", activation="sigmoid", input_dim=10))
sequential.compile(optimizer="Adamax", metrics=["accuracy"], loss="binary_crossentropy")
model_history = sequential.fit(X_train, Y_train, validation_split=0.3, batch_size=50, epochs=100)
print(model_history.history["accuracy"])
# # summarize history for accuracy
# from matplotlib  import pyplot as plt
# plt.plot(model_history.history['acc'])
# plt.plot(model_history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# # summarize history for loss
# plt.plot(model_history.history['loss'])
# plt.plot(model_history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# # Part 3 - Making the predictions and evaluating the model
#
# # Predicting the Test set results
# y_pred = sequential.predict(X_test)
# y_pred = (y_pred > 0.5)
