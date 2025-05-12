import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def build_model(input_shape=(4, 1)):
    model = Sequential([
        LSTM(32, input_shape=input_shape),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Example use
if __name__ == "__main__":
    model = build_model()
    model.summary()
