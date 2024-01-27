# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.sequence import pad_sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense


# Read stock data
f = open('./stock_dataset.csv')  # Open the stock dataset file
df = pd.read_csv(f)  # Read the stock data in CSV format, returning a DataFrame
df.head()  # Display the first few rows of the dataset


data = df.iloc[:, 2:10].values  # Select columns 3 to 10 from the stock dataset as feature data
data = data.astype('float32')  # Convert the data type to float32
data


def get_train_data(time_step=20, train_begin=0, train_end=5800):
    data_train = data[train_begin:train_end]  # Divide the training dataset
    y_ = data_train[time_step:train_end, 7]  # Obtain the 8th column in the dataset as y-value labels
    normalize_train_data = data_train
    train_x, train_y = [], []  # Training set x and y
    for i in range(len(normalized_train_data) - time_step):
        x = normalized_train_data[i:i + time_step, :7]  # Use the past 20 days' 7-dimensional data as features
        y = normalized_train_data[i + time_step:i + 1 + time_step, 7]  # Get the next day's stock price as a label
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    mean, std = np.mean(train_y, axis=0), np.std(train_y, axis=0)  # Calculate the mean and standard deviation for each column
    train_x, train_y, mean, std = np.array(train_x), np.array(train_y), np.array(mean), np.array(std)
    train_x = (train_x - np.mean(train_x, axis=0) / np.std(train_x, axis=0))  # Feature normalization
    train_y = (train_y - np.mean(train_y, axis=0) / np.std(train_y, axis=0))  # Label normalization
    return train_x, train_y, mean, std, y_

# Obtain training set data and their labels
print(data.shape)
# Set time step to 20, training set starts at 2000 and ends at 5800
train_x, train_y, _, _, _ = get_train_data(20, 2000, 5800)  
# Test set starts at 5800 and ends at the length of the dataset
test_x, test_y, test_mean, test_std, test_y = get_train_data(20, 5800, len(data))  


model = Sequential()  # Initialize a sequential model

model.add(LSTM(128, input_shape=(20, 7), dropout=0.2, recurrent_dropout=0.2,
               return_sequences=True))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
model.add(Dense(1))  # Add a fully connected layer

model.compile(loss='mse', optimizer='Adam', metrics=['mse'])  # Compile the model, specifying mean squared error (MSE) as the loss function and Adam as the optimizer

print(model.summary())

model = Sequential()
model.add(LSTM(256, input_shape=(20, 7), dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
model.add(Dense(1))

model = Sequential()
model.add(LSTM(128, input_shape=(20, 7), dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
model.add(Dense(1))


history = model.fit(x=train_x, y=train_y,
                    batch_size=60,  # Specify the batch size
                    epochs=400,  # Specify the number of iterations
                    validation_data=(test_x, test_y))  # Validate using the test set


x, y, x_mean, y_std, y_label = get_train_data(20, 5800, 6000)  # Obtain data for prediction
print(x.shape)
print(x)
print(x_mean.shape)
print(y_std.shape)
print(y_label.shape)

result = model.predict(x, batch_size=1)  # Use the trained model for prediction
print(result.shape)
print(result)

result = np.array(result) * y_std + x_mean  # Convert standardized predicted results to the original data range
print(result.shape)

result = np.squeeze(result)  # Remove redundant dimensions
print(y_std, x_mean)
print(y_label)
print(result)


# Visualizing actual and predicted values
plt.figure()
data_list = [str(i) for i in range(0, 180)]
plt.plot(list(range(len(data_list[0:180]))), y_label[0:180], color='r')
plt.plot(list(range(len(data_list))), result, color='b')
plt.xlabel('date')
plt.ylabel('stock index')
plt.show()