import keras
from keras.layers import *

from data_preparation import extract_notes_from_all_files


def create_batches(pieces):
    for piece in pieces:
        X = np.zeros((1, 64))
        y = np.zeros((1, 64, 128))
        for i in range(64):
            X[0, i] = piece[i]
            if i == 63:
                y[0, i, piece[i]] = 1
            else:
                y[0, i, piece[i+1]] = 1
        yield X, y


def create_train_model():
    model = keras.Sequential()
    model.add(Embedding(input_dim=128, output_dim=512, batch_input_shape=(1, 64)))
    model.add(LSTM(units=256, return_sequences=True, stateful=True))
    model.add(TimeDistributed(Dense(128)))
    model.add(Activation("softmax"))
    return model


def create_prediction_model():
    model = keras.Sequential()
    model.add(Embedding(input_dim=128, output_dim=512, batch_input_shape=(1, 1)))
    model.add(LSTM(units=256, return_sequences=True, stateful=True))
    model.add(TimeDistributed(Dense(128)))
    model.add(Activation("softmax"))
    return model


def train(pieces, num_epochs):
    model = create_train_model()
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    for epoch in range(num_epochs):
        print("Epoch: {}".format(epoch+1))
        loss, accuracy = 0, 0
        for i, (x, y) in enumerate(create_batches(pieces)):
            loss, accuracy = model.train_on_batch(x, y)
        print("Loss: {}, Accuracy: {}". format(loss, accuracy))

    return model


pieces = extract_notes_from_all_files("data/midi_files")
pieces = [piece[:64] for piece in pieces]
model = train(pieces, 50)
model.save_weights("model.h5")
model = create_prediction_model()
model.load_weights("model.h5")
generated = [60]

for _ in range(64):
    batch = np.zeros((1, 1))
    batch[0, 0] = 60
    predicted_probs = model.predict_on_batch(batch).ravel()
    sample = np.random.choice(range(128), size=1, p=predicted_probs)
    generated.append(sample[0])

print(generated)
