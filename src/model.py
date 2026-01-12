from tensorflow import keras
from tensorflow.keras import layers

def create_mlp(input_dim):
    """
    Builds a simple but powerful MLP model for gender classification.
    """

    model = keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=(input_dim,)))

    # Hidden Layer 1

    model.add(layers.Dense(512, activation='relu'))
    # Hidden Layer 2
    model.add(layers.Dense(256, activation='relu'))

    # Hidden Layer 3
    model.add(layers.Dense(128, activation='relu'))

 
    # Output layer (1 neuron → binary classification)
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())
    return model
