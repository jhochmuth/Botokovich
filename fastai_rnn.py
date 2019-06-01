from fastai.text import *

import numpy as np

import pandas as pd

# Get piece sequences
sequences = np.load("data/pieces.npy", allow_pickle=True)

# Shorten sequences to length of shortest sequence
seq_length = len(min(sequences, key=len))
pieces = [piece[:seq_length] for piece in sequences]

# Convert all ints to strings
pieces = [list(map(str, piece)) for piece in pieces]

# Merge so that each sequence is represented by a single string
str_seqs = list()
for piece in pieces:
    str_seqs.append(' '.join(piece))

# Create dataframes for train and validation sets
train_val_ind = int(len(pieces) * .8)
train, val = pd.DataFrame(), pd.DataFrame()
train['seq'] = str_seqs[:train_val_ind]
val['seq'] = str_seqs[train_val_ind:]

# Create databunch
data_lm = TextLMDataBunch.from_df("data", train_df=train, valid_df=val, text_cols='seq')
data_bunch = data_lm.create(data_lm.train_ds, data_lm.valid_ds)

# Create model
learner = language_model_learner(data=data_bunch, arch=AWD_LSTM, pretrained=False)

# Train model
learner.fit(epochs=10)
