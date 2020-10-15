import pickle

from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, LSTM
from keras import backend as K

from old import *


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


data = fetch_from_api(20000)

contextualized_data = calculate_derived_features(data)

scaled_data, scalers = scale_features(
    contextualized_data,
    ["high", "low", "open", "close", "volume", "time since last inversion"],
    ["delta low/high", "delta open/close", "delta previous/current"]
)

CONTEXT = 60

X, Y = create_training_data(scaled_data.to_numpy(), CONTEXT)

trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.8)

features = scaled_data.shape[1]

model = Sequential()
model.add(LSTM(128, input_dim=features, input_length=CONTEXT))
model.add(Dense(5))
model.compile(loss=root_mean_squared_error, optimizer='adam')

model.fit(trainX, trainY, epochs=20, batch_size=60, verbose=2, shuffle=True)

model.save(f"models/saved_model")
with open("../models/model_scalers.pickle", 'wb') as f:
    pickle.dump(scalers, f)