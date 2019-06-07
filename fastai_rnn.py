"""Uses language models architecture from the fastai library.

"""
from fastai.text import *

import numpy as np

import pandas as pd


config = awd_lstm_lm_config.copy()
config['n_hid'] = 100
#config['weight_p'] = .1
#config['hidden_p'] = .05

epochs = 50
bs = 5
lr = .01


def collect_sequences(filename):
    sequences = np.load(filename)
    return sequences


def truncate_sequences(sequences):
    seq_length = len(min(sequences, key=len))
    sequences = [sequence[:seq_length] for sequence in sequences]
    return sequences


def convert_lists_to_strings(sequences):
    sequences = [list(map(str, sequence)) for sequence in sequences]
    str_seqs = list()
    for sequence in sequences:
        str_seqs.append(' '.join(sequence))
    return str_seqs


def create_train_val_sets(str_seqs):
    train_val_ind = int(len(str_seqs) * .8)
    train, val = pd.DataFrame(), pd.DataFrame()
    train['seq'] = str_seqs[:train_val_ind]
    val['seq'] = str_seqs[train_val_ind:]
    return train, val


def create_databunch(train, val, bs):
    tok = Tokenizer(pre_rules=list(), post_rules=list(), special_cases=[UNK])
    data_lm = TextLMDataBunch.from_df("data", train_df=train, valid_df=val, text_cols='seq', tokenizer=tok, bs=bs)
    return data_lm


def create_model_and_train(data_lm, config, epochs, output_file="fastai_rnn"):
    learner = language_model_learner(data=data_lm, arch=AWD_LSTM, pretrained=False, config=config)
    learner.fit(epochs=epochs, lr=lr)
    learner.load()
    learner.save(output_file)
    return learner


def predict(learner, start, length):
    start = "xxbos " + start
    return learner.predict(start, n_words=length * 12)


def main():
    # The following code is used for training and predicting using notes without timing.
    """
    sequences = collect_sequences("data/train_sequences/cello_sequences.npy")
    str_seqs = convert_lists_to_strings(sequences)
    train, val = create_train_val_sets(str_seqs)
    data_lm = create_databunch(train, val, bs)
    learner = create_model_and_train(data_lm, config, epochs)
    print(predict(learner, "60", 10))
    """

    # The following code is used for training and predicting using chord_sequences
    """
    sequences = np.load("data/train_sequences/chord_sequences.npy", allow_pickle=True)
    train, val = create_train_val_sets(sequences)
    data_lm = create_databunch(train, val, bs)
    learner = create_model_and_train(data_lm, config, epochs)

    test_sequence = ["0" for _ in range(48)]
    test_sequence[12], test_sequence[24], test_sequence[28] = "1", "1", "1"
    test_sequence = "".join(test_sequence)
    print(predict(learner, test_sequence, 50))
    """

    sequences = np.load("data/train_sequences/note_sequences.npy", allow_pickle=True)
    train, val = create_train_val_sets(sequences)
    data_lm = create_databunch(train, val, bs)
    learner = create_model_and_train(data_lm, config, epochs)
    test_sequence = "xxbos 12 24 31 step step step step step step"
    predict(learner, test_sequence, 100)


if __name__ == "__main__":
    main()
