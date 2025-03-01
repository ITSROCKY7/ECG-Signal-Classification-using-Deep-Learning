!pip install tensorflow numpy matplotlib wfdb
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt
import wfdb
# Load ECG signal (MIT-BIH Arrhythmia Database sample)
record = wfdb.rdrecord("mitdb/100", sampto=3000)  # Load first 3000 samples
ecg_signal = record.p_signal[:, 0]  # Take Lead I ECG

plt.plot(ecg_signal)
plt.title("Sample ECG Signal")
plt.show()
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(3000, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
