import pickle

import keras
from keras import backend as K

from old import *

CONTEXT = 60
TIMESTEPS = 6000

data = fetch_from_api(CONTEXT, check_cache=False)

contextualized_data = calculate_derived_features(data)

scalers = pickle.load(open("../models/model_scalers.pickle", 'rb'))

scaled_data = rescale(contextualized_data, scalers)

X = scaled_data.to_numpy().reshape((1, CONTEXT, 9))


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


model = keras.models.load_model("../models/saved_model", custom_objects={"root_mean_squared_error": root_mean_squared_error})

for iteration in range(100):
    predicted_Y = model.predict(X)
    X = np.append(X, predicted_Y)

# plt.axes()
#
# plt.plot([dp[3] for dp in X[0]])
# plt.plot(predicted_Y[3])
# plt.grid()
# plt.show()

pass
