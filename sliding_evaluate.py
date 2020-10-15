import pickle

from matplotlib import pyplot as plt
import pandas as pd
from keras import backend as K
import keras

import data_processing as dp

CONTEXT = 60

raw_data = dp.load_from_poloniex(CONTEXT + 1)
derived = dp.generate_derived(raw_data)

with open("models/model_scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

scaled = dp.rescale(derived, scalers)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


model = keras.models.load_model("models/saved_model", custom_objects={"root_mean_squared_error": root_mean_squared_error})

for i in range(100):
    X = scaled[-60:].to_numpy().reshape((1, CONTEXT, 9))
    predicted = model.predict(X, batch_size=60)[0]
    new_raw = dp.dataframe_from_raw(predicted)
    new_descaled = dp.descale(new_raw, scalers)

    raw_data = raw_data.append(new_descaled)
    derived = dp.generate_derived(raw_data)
    scaled = dp.rescale(derived, scalers)
    pass

plt.axes()

plt.plot(scaled["close"])
plt.plot(range(61, 2000), predicted_Y[2])
plt.grid()
plt.show()

pass