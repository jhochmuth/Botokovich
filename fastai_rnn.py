from fastai.text import *

import numpy as np

import pandas as pd


config = awd_lstm_lm_config.copy()
config['emb_sz'] = 1
config['n_hid'] = 500
config['weight_p'] = .1
config['hidden_p'] = .05

epochs = 50
bs = 10
lr = .01


def collect_sequences(filename):
    sequences = np.load(filename, allow_pickle=True)
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


def create_model_and_train(data_lm, config, epochs):
    learner = language_model_learner(data=data_lm, arch=AWD_LSTM, pretrained=False, config=config)
    learner.fit(epochs=epochs, lr=lr)
    learner.save("awd_lstm")
    return learner


def predict(learner, start, length):
    start = "xxbos " + start
    return learner.predict(start, n_words=length)


def main():
    """
    sequences = collect_sequences("data/cello_sequences.npy")
    sequences = truncate_sequences(sequences)
    str_seqs = convert_lists_to_strings(sequences)
    train, val = create_train_val_sets(str_seqs)
    data_lm = create_databunch(train, val, bs)
    learner = create_model_and_train(data_lm, config, epochs)
    print(predict(learner, "60", 10))
    """
    sequences = np.load("data/chord_sequences.npy", allow_pickle=True)
    sequences = truncate_sequences(sequences)
    train, val = create_train_val_sets(sequences)
    data_lm = create_databunch(train, val, bs)
    learner = create_model_and_train(data_lm, config, epochs)

    test_sequence = ["0" for _ in range(48)]
    test_sequence[12], test_sequence[24], test_sequence[28] = "1", "1", "1"
    test_sequence = "".join(test_sequence)
    print(predict(learner, test_sequence, 50))


if __name__ == "__main__":
    main()
