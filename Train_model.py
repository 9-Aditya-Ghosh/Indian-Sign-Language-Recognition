import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Load dataset
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

X = np.array(data['data'])
y = to_categorical(np.array(data['labels']))

# Padding sequences
max_length = max(len(seq) for seq in X)
X_padded = np.zeros((len(X), max_length))
for i, seq in enumerate(X):
    X_padded[i, :len(seq)] = seq

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, stratify=y)

# Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_padded.shape[1], 1)),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

# Save model
model.save('models/gesture_model.h5')
