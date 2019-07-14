"""Uses keras RNN architecture.

"""
import keras
from keras.layers import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import numpy as np


def create_tokens2(filename):
    sequences = np.load(filename)
    num_sequences = len(sequences)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    sequences = tokenizer.texts_to_sequences(sequences)
    sequences = pad_sequences(sequences, 300)
    vocab_size = len(tokenizer.word_index) + 1
    X, y = sequences[:][:-1], sequences[:][-1]
    X = np.reshape(X, (num_sequences-1, 300, 1))
    y = to_categorical(y, num_classes=vocab_size)
    seq_length = 300
    return vocab_size, seq_length, X, y


def create_tokens(filename):
    sequences = np.load(filename)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    X = list()
    y = list()

    for seq in sequences:
        seq_arr = seq.split()
        for length in range(300-1):
            X.append(seq_arr[:length + 1])
            y.append(seq_arr[length + 1])

    X = tokenizer.texts_to_sequences(X)
    y = tokenizer.texts_to_sequences(y)
    X = pad_sequences(X, 300)
    vocab_size = len(tokenizer.word_index) + 1
    X = np.reshape(X, (len(X), 300, 1))
    y = to_categorical(y, num_classes=vocab_size)
    seq_length = 300
    return vocab_size, seq_length, X, y


def create_train_model(vocab_size, seq_length):
    model = keras.Sequential()
    #model.add(Embedding(input_dim=vocab_size, output_dim=512, input_shape=(300, 1)))
    model.add(LSTM(256, input_shape=(seq_length, 1), return_sequences=True))
    model.add(LSTM(units=256, return_sequences=True, stateful=False))
    #model.add(Dropout(rate=.2))
    model.add(LSTM(units=256, stateful=False))
    model.add(Dense(vocab_size))
    model.add(Activation("softmax"))
    return model


def create_prediction_model(vocab_size):
    model = keras.Sequential()
    #model.add(Embedding(input_dim=vocab_size, output_dim=512, input_shape=(300, 1)))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(units=256, return_sequences=True, stateful=False))
    #model.add(Dropout(rate=.2))
    model.add(LSTM(units=256, stateful=False))
    model.add(Dense(vocab_size))
    model.add(Activation("softmax"))
    return model


def train(X, y, vocab_size, num_epochs, seq_length):
    model = create_train_model(vocab_size, seq_length)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X, y, batch_size=32, epochs=num_epochs)

    model.save_weights("data/keras_rnn.h5")


def generate_sequence(seq_length, start_note, model_weights_file):
    model = create_prediction_model()
    model.load_weights(model_weights_file)
    seq = [60]

    for _ in range(seq_length - 1):
        batch = np.zeros((1, 1))
        batch[0, 0] = start_note
        predicted_probs = model.predict_on_batch(batch).ravel()
        sample = np.random.choice(range(128), size=1, p=predicted_probs)
        seq.append(sample[0])

    return seq


def train_and_generate(seq_file, num_epochs):
    vocab_size, seq_length, X, y = create_tokens("data/train_sequences/all_note_sequences.npy")
    train(X, y, vocab_size, num_epochs, seq_length)
    #generated = generate_sequence(seq_length, 48, "models/keras_only_notes_model.h5")
    #print(generated)


def main():
    train_and_generate("data/all_note_sequences.npy", 50)


if __name__ == "__main__":
    main()
