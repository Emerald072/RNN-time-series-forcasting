import sys
from loguru import logger

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from data_preparation import split_train_test
from data_preparation import create_sequences
from data_preparation import create_dataloader
from models.RNN import get_rnn_model
from utils import data_plot
import math
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
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
logger.info(f"Train/Test Split")
data = split_train_test(df=df, training_ratio=0.75, price_type='Open')

logger.info(f"Normalizing Data")
# 2. Normalization of Training and Testing set - SEPARATELY
scaler = MinMaxScaler(feature_range=(0, 1))

# Scaling dataset
data["training_data"]= scaler.fit_transform(data["training_data"])
print(data["training_data"][:10])

# Normalizing values between 0 and 1
data["testing_data"] = scaler.fit_transform(data["testing_data"])
print(data["testing_data"][:10])

# We split and normalized the data, now we need to create
# sequences and the corresponding labels
logger.info(f"Creating Time Series Sequences")
x_train, y_train = create_sequences(data["training_data"], sequence_length=50)
x_test, y_test = create_sequences(data["testing_data"], sequence_length=30)

# Define Value
model_type = "GRU"
input_size = 1
num_layers = 3  # Increased number of layers
hidden_size = 128  # Increased number of hidden units
output_size = 1
dropout = 0.3  # Added dropout for regularization
batch_size = 128

# Prepare the model and put it on device (gpu or cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_rnn_model(model_type,input_size, hidden_size, num_layers, dropout)
model = model.to(device)

# Defining my loss/cost function
loss_fn = nn.MSELoss(reduction='mean')

# Defining my optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Creating DataLoaders to provide batch of data of size `batch_size`
train_loader = create_dataloader(x_train, y_train, batch_size=batch_size, shuffle=True)
test_loader = create_dataloader(x_test, y_test, batch_size=batch_size, shuffle=True)

# The training loop
num_epochs = 100  # Increased number of epochs
train_hist = []
test_hist = []

logger.info(f"Starting Training...")
for epoch in range(num_epochs):
    total_loss = 0.0
    model.train()
    print(f"Epoch {epoch+1}/{num_epochs}")
    # print("Epoch {}/{}".format(epoch + 1, num_epochs))
    for batch_x, batch_y in tqdm(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        predictions = model(batch_x)
        loss = loss_fn(predictions, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    train_hist.append(average_loss)

    #Evaluate on test data
    model.eval()
    with torch.no_grad():
        total_test_loss = 0.0

        for batch_X_test, batch_y_test in test_loader:
            batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
            predictions_test = model(batch_X_test)
            test_loss = loss_fn(predictions_test, batch_y_test)

            total_test_loss += test_loss.item()

        average_test_loss = total_test_loss / len(test_loader)
        test_hist.append(average_test_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}')

logger.info(f"..finished!")
x = np.linspace(1,num_epochs,num_epochs)
plt.plot(x,train_hist,scalex=True, label="Training loss")
plt.plot(x, test_hist, label="Test loss")
plt.legend()
plt.show()