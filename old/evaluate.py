import pickle

import keras
import matplotlib.pyplot as plt
import pandas
from keras import backend as K

CONTEXT = 60
TIMESTEPS = 6000

data = fetch_from_api(TIMESTEPS, check_cache=False)

contextualized_data = calculate_derived_features(data)

scalers = pickle.load(open("../models/model_scalers.pickle", 'rb'))

scaled_data = rescale(contextualized_data, scalers)

X, Y = create_training_data(scaled_data.to_numpy(), CONTEXT)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


model = keras.models.load_model("../models/saved_model", custom_objects={"root_mean_squared_error": root_mean_squared_error})

predicted_Y = pandas.DataFrame(model.predict(X))

plt.axes()

plt.plot(scaled_data["close"])
plt.plot(predicted_Y[2])
plt.grid()
plt.show()

pass
