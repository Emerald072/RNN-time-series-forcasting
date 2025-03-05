import math
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def split_train_test(df, training_ratio=0.8, price_type='close'):
    #Only takes Open Price!
    #TODO Extend to take other price

    training_data_len = math.ceil(len(df) * training_ratio)
    print(training_data_len)

    # Splitting the dataset
    train_data = df[:training_data_len].iloc[:, :4]
    test_data = df[training_data_len:].iloc[:, :4]

    if price_type.lower() == 'close':
        train_data = train_data.Close.values
        test_data = test_data.Close.values
    elif price_type.lower() == 'open':

        train_data = train_data.Open.values
        test_data = test_data.Open.values
    else:
        raise NotImplementedError
    train_data = np.reshape(train_data, (-1, 1))
    test_data = np.reshape(test_data, (-1, 1))
    data = {"training_data":train_data, "testing_data":test_data} # data["training_data"]
    return data


def create_sequences(data, sequence_length):
    """ Create sequences for time series prediction

    :param data: Corresponding data
    :param sequence_length: The length of the sequence
    :return: x, y
    """
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])  # Predicting the value right after the sequence
    x, y = np.array(x), np.array(y)

    # Convert data to PyTorch tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    print(x.shape, y.shape)
    return x, y


def create_dataloader(x, y, batch_size, shuffle=True):
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
