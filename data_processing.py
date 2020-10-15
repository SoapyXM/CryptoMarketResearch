import pickle
from datetime import datetime as dt
from typing import Dict

import requests

import pandas as pd
import numpy as np
from numba import jit
from sklearn.preprocessing import MinMaxScaler

URL = "https://poloniex.com/public"
TIMESTEP = 1800


def load_from_poloniex(timesteps: int) -> pd.DataFrame:
    parameters = {
        "command": "returnChartData",
        "currencyPair": "USDT_ETH",
        "period": TIMESTEP,
        "start": (int(dt.now().timestamp() / TIMESTEP) - timesteps + 1) * TIMESTEP,
        "end": 99999999999
    }

    json = requests.get(url=URL, params=parameters).content

    data = pd.read_json(json).iloc[:, 1:6]

    print(f"Loaded {timesteps} timesteps.")

    return data


def dataframe_from_raw(raw: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"high": [raw[0]], "low": [raw[1]], "open": [raw[2]], "close": [raw[3]], "volume": [raw[4]]})


@jit
def calculate_inversion_times(deltas: np.ndarray) -> np.ndarray:
    inversions = np.zeros(deltas.size)

    time_since_last_inversion = 0
    last_sign = np.sign(deltas[0])
    for i in range(1, deltas.size):
        sign = np.sign(deltas[i])
        if sign != last_sign:
            time_since_last_inversion = 0
        else:
            time_since_last_inversion += 1
        last_sign = sign
        inversions[i] = time_since_last_inversion

    return inversions


def generate_derived(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["delta open/close"] = data["close"] - data["open"]
    data["delta low/high"] = data["high"] - data["low"]
    data["delta previous/current"] = data["close"].diff()
    data["time since last inversion"] = calculate_inversion_times(data["delta previous/current"].to_numpy())

    data.drop(0, inplace=True)
    data.reset_index(0, inplace=True, drop=True)

    return data


def scale_features(data: pd.DataFrame, standard, signed) -> (pd.DataFrame, Dict[str, MinMaxScaler]):
    data = data.copy()

    scalers = {}

    for col in standard:
        scaler = MinMaxScaler()
        data_fixed = data[col].to_numpy().reshape(-1, 1)
        scaler.fit(data_fixed)
        data[col] = scaler.transform(data_fixed)
        scalers[col] = scaler

    for col in signed:
        scaler = MinMaxScaler((-1, 1))
        data_fixed = data[col].to_numpy().reshape(-1, 1)
        scaler.fit(data_fixed)
        data[col] = scaler.transform(data_fixed)
        scalers[col] = scaler

    return data, scalers


def rescale(data: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    data = data.copy()

    for col in scalers.keys():
        data_fixed = data[col].to_numpy().reshape(-1, 1)
        data[col] = scalers[col].transform(data_fixed)

    return data


def descale(data: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    data = data.copy()

    for col in scalers.keys():
        try:
            data_fixed = data[col].to_numpy().reshape(-1, 1)
            data[col] = scalers[col].inverse_transform(data_fixed)
        except KeyError:
            pass

    return data


def create_training_data(data: np.ndarray, context: int):
    features = data.shape[1]
    datapoint_count = data.shape[0]
    X = np.ndarray((datapoint_count - context, context, features))
    Y = np.ndarray((datapoint_count - context, 5))

    for i in range(context, datapoint_count):
        X[i - context] = data[i - context:i, :]
        Y[i - context] = data[i, :5]
    return X, Y


def main():
    with open("data/unscaled_data.json", 'r') as f:
        contents = f.read()
        raw_data = pd.read_json(contents).iloc[:, 1:6]
    derived = generate_derived(raw_data)
    scaled_data, scalers = scale_features(
        derived,
        ["high", "low", "open", "close", "volume", "time since last inversion"],
        ["delta low/high", "delta open/close", "delta previous/current"]
    )
    scaled_data.to_csv("data/scaled_data.csv", index=False)
    with open("data/data_scaler.pkl", 'wb') as f:
        pickle.dump(scalers, f)
    print("Done!")


if __name__ == "__main__":
    main()
