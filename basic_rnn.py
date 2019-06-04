import keras
from keras.layers import *

import numpy as np


def create_batches(pieces):
    for piece in pieces:
        X = np.zeros((1, len(piece)))
        y = np.zeros((1, len(piece), 128))
        for i in range(len(piece)):
            X[0, i] = piece[i]
            if i == len(piece) - 1:
                y[0, i, piece[i]] = 1
            else:
                y[0, i, piece[i+1]] = 1
        yield X, y


def create_train_model(seq_length):
    model = keras.Sequential()
    model.add(Embedding(input_dim=128, output_dim=512, batch_input_shape=(1, seq_length)))
    model.add(LSTM(units=256, return_sequences=True, stateful=True))
    model.add(Dropout(rate=.2))
    model.add(LSTM(units=256, return_sequences=True, stateful=True))
    model.add(TimeDistributed(Dense(128)))
    model.add(Activation("softmax"))
    return model


def create_prediction_model():
    model = keras.Sequential()
    model.add(Embedding(input_dim=128, output_dim=512, batch_input_shape=(1, 1)))
    model.add(LSTM(units=256, return_sequences=True, stateful=True))
    model.add(Dropout(rate=.2))
    model.add(LSTM(units=256, return_sequences=True, stateful=True))
    model.add(TimeDistributed(Dense(128)))
    model.add(Activation("softmax"))
    return model


def train(pieces, num_epochs, seq_length):
    model = create_train_model(seq_length)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    for epoch in range(num_epochs):
        print("Epoch: {}".format(epoch+1))
        loss, accuracy = 0, 0
        for i, (x, y) in enumerate(create_batches(pieces)):
            loss, accuracy = model.train_on_batch(x, y)
        print("Loss: {}, Accuracy: {}". format(loss, accuracy))

    model.save_weights("data/model.h5")


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


def train_and_generate_only_notes(seq_file, num_epochs):
    sequences = np.load(seq_file, allow_pickle=True)
    seq_length = len(min(sequences, key=len))
    pieces = [piece[:seq_length] for piece in sequences]
    train(pieces, num_epochs, seq_length)
    generated = generate_sequence(seq_length, 48, "data/model.h5")
    print(generated)


def main():
    train_and_generate_only_notes("data/note_sequences.npy", 50)


if __name__ == "__main__":
    main()
