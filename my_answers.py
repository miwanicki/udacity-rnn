import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

from string import ascii_lowercase


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    try:
        assert(window_size < len(series))

        X = []

        for it in range(len(series)-window_size):
            X.append(series[it:it+window_size])

        y = series[window_size:]

        # reshape each
        X = np.asarray(X)
        X.shape = (np.shape(X)[0:2])
        y = np.asarray(y)
        y.shape = (len(y),1)

        return X,y
    except AssertionError:
        print('Windows size is greater than or equal to length of the series')
        return None,None

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):

    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1, activation='tanh'))

    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    acceptable = ascii_lowercase + ''.join(punctuation) + ' '

    text = ''.join([x for x in text if x in acceptable])

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in np.arange(0, len(text)-window_size, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])

    return inputs,outputs

# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):

    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))

    return model
