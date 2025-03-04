import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from tqdm.notebook import tqdm

import torch
from data_preparation import split_train_test
from utils import data_plot
import math
from sklearn.preprocessing import MinMaxScaler

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

Colour_Palette = ['#01BEFE', '#FF7D00', '#FFDD00', '#FF006D', '#ADFF02', '#8F00FF']
sns.set_palette(sns.color_palette(Colour_Palette))

tqdm.pandas()

import yfinance as yf
from datetime import date

end_date = date.today().strftime("%Y-%m-%d")
start_date = '2000-01-01'

# Creating Yahoo Finance DataFrame
df = yf.download('AAPL', start=start_date, end=end_date)

# Inspect the data
print(df.columns)

# Plot the data
data_plot(df)


# Price at closing is index 0
# Price at High is index 1
# Price at Low is index 2
# Price at Open is index 3
# For example to get only prices at opening: df.iloc[0, 3]
print(df.iloc[0, :4])

# 1. Train/Test split
data = split_train_test(df=df, training_ratio=0.75, price_type='Open')


# 2. Normalization of Training and Testing set - SEPARATELY
scaler = MinMaxScaler(feature_range=(0, 1))
# Scaling dataset
data["training_data"]= scaler.fit_transform(data["training_data"])
print(data["training_data"][:10])

# Normalizing values between 0 and 1
data["testing_data"] = scaler.fit_transform(data["testing_data"])
print(data["testing_data"][:10])
