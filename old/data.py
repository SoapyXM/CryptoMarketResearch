import os
from datetime import datetime as dt
from typing import List, Dict

import numpy as np
import pandas as pd
import requests
from numba import jit
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

URL = "https://poloniex.com/public"
TIMESTEP = 1800


def fetch_from_api(timesteps: int, check_cache=False) -> pd.DataFrame:
    parameters = {
        "command": "returnChartData",
        "currencyPair": "USDT_ETH",
        "period": TIMESTEP,
        "start": int(dt.now().timestamp()) - TIMESTEP * (timesteps),
        "end": 99999999999
    }

    json = requests.get(url=URL, params=parameters).content

    data = pd.read_json(json).iloc[:, 1:6]

    print(f"Loaded {timesteps} timesteps.")

    return data





def calculate_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["delta open/close"] = data["close"] - data["open"]
    data["delta low/high"] = data["high"] - data["low"]
    data["delta previous/current"] = data["close"].diff()
    data["time since last inversion"] = calculate_inversion_times(data["delta previous/current"].to_numpy())

    data.drop(0, inplace=True)
    data.reset_index(0, inplace=True, drop=True)

    return data


def scale_features(data: pd.DataFrame, standard, signed) -> (DataFrame, Dict[str, MinMaxScaler], Dict[str, MinMaxScaler]):
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



