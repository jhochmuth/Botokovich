"""Uses keras RNN architecture.

"""
import keras
from keras.layers import *
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import numpy as np


def create_tokens(filename):
    sequences = np.load(filename)
    sequences = sequences.split()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    sequences = tokenizer.texts_to_sequences(sequences)
    vocab_size = len(tokenizer.word_index) + 1
    X, y = sequences[:, :-1], sequences[:, -1]
    y = to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1]
    return vocab_size, seq_length, X, y


def create_train_model(vocab_size, seq_length):
    model = keras.Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=512, input_length=seq_length))
    model.add(LSTM(units=256, return_sequences=True, stateful=True))
    model.add(Dropout(rate=.2))
    model.add(LSTM(units=256, return_sequences=True, stateful=True))
    model.add(TimeDistributed(Dense(128)))
    model.add(Activation("softmax"))
    return model


def create_prediction_model(vocab_size):
    model = keras.Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=512))
    model.add(LSTM(units=256, return_sequences=True, stateful=True))
    model.add(Dropout(rate=.2))
    model.add(LSTM(units=256, return_sequences=True, stateful=True))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation("softmax"))
    return model


def train(X, y, num_epochs, seq_length):
    model = create_train_model(seq_length)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X, y, batch_size=128, epochs=num_epochs)

    model.save_weights("data/keras_only_notes_model.h5")


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
    train(X, y, num_epochs, seq_length)
    #generated = generate_sequence(seq_length, 48, "models/keras_only_notes_model.h5")
    #print(generated)


def main():
    train_and_generate("data/all_note_sequences.npy", 50)


if __name__ == "__main__":
    main()
