
'''
Example script to train a model to generate text from a text file.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
import numpy as np
import random
import math
import os
import codecs
import collections
from six.moves import cPickle


data_dir = 'data/Artistes_et_Phalanges-David_Campion'
save_dir = 'save'
rnn_size = 32 #128
batch_size = 64
seq_length = 5 #15
num_epochs = 8
learning_rate = 0.001
sequences_step = 1
split_number = 10

input_file = os.path.join(data_dir, "input.txt")
vocab_file = os.path.join(save_dir, "words_vocab.pkl")

with codecs.open(input_file, "r", encoding=None) as f:
    data = f.read()

x_text = data.split()

word_counts = collections.Counter(x_text)

vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))

vocab = {x: i for i, x in enumerate(vocabulary_inv)}
words = [x[0] for x in word_counts.most_common()]

vocab_size = len(words)

with open(os.path.join(vocab_file), 'wb') as f:
    cPickle.dump((words, vocab, vocabulary_inv), f)

sequences = []
next_words = []
for i in range(0, len(x_text) - seq_length, sequences_step):
    sequences.append(x_text[i: i + seq_length])
    next_words.append(x_text[i + seq_length])

print('nb sequences:', len(sequences))

print('init LSTM model')
model = Sequential()
model.add(LSTM(rnn_size, input_shape=(seq_length, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

indexes = list(range(len(sequences)))
nb_to_draw = math.floor(len(sequences)/split_number)
nb_draw = math.ceil(len(sequences)/nb_to_draw)

print('drawing ', nb_draw, ' time a maximum of ', nb_to_draw, ' elements')

for epoch in range(num_epochs):
    print(epoch, '/', num_epochs, ' - starting')
    random.shuffle(indexes)
    draw = 0
    idx = 0
    max_idx = 0
    for d in range(nb_draw):
        print(epoch, '/', num_epochs, ' - ', d, '/', nb_draw, ' - preparing data')
        max_idx = min(max_idx + nb_to_draw, len(indexes))
        some_sequences = []
        some_next_words = []
        while idx < max_idx:
            some_sequences.append(sequences[indexes[idx]])
            some_next_words.append(next_words[indexes[idx]])
            idx += 1
        print(epoch, '/', num_epochs, ' - ', d, '/', nb_draw, ' - learning on ', len(some_sequences), ' sentences')
        X = np.zeros((len(some_sequences), seq_length, vocab_size), dtype=np.bool)
        y = np.zeros((len(some_sequences), vocab_size), dtype=np.bool)
        for i, sentence in enumerate(some_sequences):
            for t, word in enumerate(sentence):
                X[i, t, vocab[word]] = 1
            y[i, vocab[some_next_words[i]]] = 1
        model.fit(X, y, batch_size=batch_size)
    model.save(save_dir + "/" + 'my_model.' + str(epoch) + '.h5')
model.save(save_dir + "/" + 'my_model.h5')
