'''
Train LSTM or GRU models on formatted data
'''

import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
from json import loads

#Read actual sentence data
data = pd.read_csv("./data/formatted_data.csv", sep="\t")

#Read and convert metadata
with open("./data/metadata.txt", "r") as f:
    english_map, spanish_map, english_vocab, spanish_vocab = f.read().split("\n")

english_map = loads(english_map)
spanish_map = loads(spanish_map)
english_vocab = int(english_vocab)
spanish_vocab = int(spanish_vocab)

#Convert formatted data from json to lists
english = data.english.to_list()
spanish = data.spanish.to_list()

for i in range(len(english)):
    english[i] = loads(english[i])
    spanish[i] = loads(spanish[i])

#Convert to arrays and reshape output
english = np.array(english)
spanish = np.array(spanish)
spanish = np.reshape(spanish, (spanish.shape[0], spanish.shape[1], 1))

#Create models for LSTM and GRU
#Embedding and output layers use vocab + 1 to account for the appended 0's
#That were not included in the original vocab calculation

#Create LSTM Model
model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Embedding(english_vocab+1, 256, batch_input_shape = [None,None]))
model1.add(tf.keras.layers.LSTM(units=1024, return_sequences = True,  stateful=False))
model1.add(tf.keras.layers.Dense(spanish_vocab+1, activation="softmax"))
model1.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model1.summary()

#Create GRU Model
model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Embedding(english_vocab+1, 256, batch_input_shape = [None,None]))
model2.add(tf.keras.layers.GRU(units=1024, return_sequences = True,  stateful=False))
model2.add(tf.keras.layers.Dense(spanish_vocab+1, activation="softmax"))
model2.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model2.summary()

checkpoint_dir = './rnn_generation'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

#Store checkpoints every 5 epochs to not run out of memory on server
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    period = 5)

#Store output accuracy logs for each epoch
logs_callback = tf.keras.callbacks.CSVLogger('./logs/log.csv', append=True, separator=";")

#If training was killed, ran out of time, or crashed, resume from latest checkpoint
resume = False
initial = 0
if resume:
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    initial = int(latest.split("_")[-1])
    model.load_weights(latest)

EPOCHS = 100

#Set training version (LSTM, GRU, GRU 512) to only train 1 model at a time
#For each model get the running time
#Model files renamed after saving to prevent accidental overrides
version = 0

start = time.perf_counter() #Starting time

#LSTM 256 batch
if version == 0:
    model1.fit(english, spanish, batch_size = 256, epochs=EPOCHS, initial_epoch=initial, callbacks=[checkpoint_callback, logs_callback])
    model1.save('./models/v0.h5')
#GRU 256 batch
elif version == 1:
    model2.fit(english, spanish, batch_size = 256, epochs=EPOCHS, initial_epoch=initial, callbacks=[checkpoint_callback, logs_callback])
    model2.save('./models/v1.h5')
#GRU 512 batch
else:
    model2.fit(english, spanish, batch_size = 512, epochs=EPOCHS, initial_epoch=initial, callbacks=[checkpoint_callback, logs_callback])
    model2.save('./models/v2.h5')

end = time.perf_counter() #Ending time

with open("./times/time.txt", "w") as f:
    f.write(str(end-start))
