import pickle

from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, LSTM
from keras import backend as K
import pandas as pd

import data_processing

CONTEXT = 60


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


scaled_data = pd.read_csv("data/scaled_data.csv")
with open("data/data_scaler.pkl", "rb") as f:
    scalers = pickle.load(f)

X, Y = data_processing.create_training_data(scaled_data.to_numpy(), CONTEXT)

trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.8)

features = scaled_data.shape[1]

model = Sequential()
model.add(LSTM(128, input_dim=features, input_length=CONTEXT))
model.add(Dense(5))
model.compile(loss=root_mean_squared_error, optimizer='adam')

model.fit(trainX, trainY, epochs=20, batch_size=60, verbose=2, shuffle=True)

model.save("models/saved_model")
with open("models/model_scalers.pkl", 'wb') as f:
    pickle.dump(scalers, f)
