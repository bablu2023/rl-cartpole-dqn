"""Deep learning model: maps a state to one Q-value per action."""
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential


def build_model(states, actions, hidden=(24, 24)):
    model = Sequential()
    # keras-rl2 feeds input shaped (window_length=1, states); flatten it to (states,)
    model.add(Flatten(input_shape=(1, states)))
    for units in hidden:
        model.add(Dense(units, activation="relu"))
    # one linear output per action = estimated Q-value (expected future reward)
    model.add(Dense(actions, activation="linear"))
    return model
