import math
import numpy as np

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
