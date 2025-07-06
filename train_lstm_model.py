import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

# Load and preprocess data
df = pd.read_csv('data/Bike-Sharing-Dataset/hour.csv')
data = df['cnt'].values.reshape(-1, 1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 24
x, y = create_sequences(data_scaled, seq_length)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=10, batch_size=32)

# Save model
if not os.path.exists('model'):
    os.makedirs('model')

model.save('model/lstm_model.h5')
print("✅ Model saved to model/lstm_model.h5")
