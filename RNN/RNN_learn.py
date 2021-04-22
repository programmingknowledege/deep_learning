import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("/home/kunal/PycharmProjects/rasa-chat/trainset.csv")
# print(data)
X = data.iloc[:, 2:].values
Y = data.iloc[:, 1:2].values
# print(X.columns)
# print(Y)
# print(data.columns)

minmaxscalar = MinMaxScaler(feature_range=(0, 1))
training_data = minmaxscalar.fit_transform(Y)
print(training_data)

# Most important part of RNN
X_train = []
Y_train = []
for i in range(60, 1258):
    X_train.append(training_data[i - 60:i, 0])
    Y_train.append(training_data[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)
print(X_train)
print(X_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# print(X_train)
# print(Y_train)
print("shhhhhhhhhhhhhhhhhappppppppppppppppe")
print(X_train.shape)

from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential

# DropOuts and Regularization are added in order to avoid the overfitting problem
regressor = Sequential()
# units is the number of neurons,
# return_sequence=TRue because muje aur layer add karna hai

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
# kitne neurons drop karne hai...isliye DropOUt use karne ka    classic number is 20 percent
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
# kitne neurons drop karne hai...isliye DropOUt use karne ka    classic number is 20 percent
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
# kitne neurons drop karne hai...isliye DropOUt use karne ka    classic number is 20 percent
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
# kitne neurons drop karne hai...isliye DropOUt use karne ka    classic number is 20 percent
regressor.add(Dropout(0.2))

# Add the output layer
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')
print(X_train.shape)
print(Y_train.shape)
# Best optimizer for RNN are rms_prop and Adam
regressor.fit(X_train, Y_train, epochs=100, batch_size=32)
# print(regressor)

test_set = pd.read_csv("/home/kunal/PycharmProjects/rasa-chat/testset.csv").iloc[:, 1:2]
# print(test_set)
final_set = pd.concat([data.iloc[:, 1:2], test_set], axis=0)

# print(final_set)

final_set = final_set[len(final_set) - len(test_set) - 60:].values
print(final_set)
# final_set=final_set.reshape(-1,1)
# print(final_set)
final_set = minmaxscalar.fit_transform(final_set)

X_test = []
Y_test = []
for i in range(60, 185):
    X_test.append(training_data[i - 60:i, 0])


X_test = np.array(X_test)
print(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape)

predict_price=regressor.predict(X_test)
print(predict_price)