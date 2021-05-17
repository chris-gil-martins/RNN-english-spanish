'''
Test the models created on the full data
'''

import tensorflow as tf
import numpy as np
import pandas as pd
from json import loads

#Read data and grab a random sample of 100 sentence pairs
data = pd.read_csv("./data/formatted_data.csv", sep="\t")
data = data.sample(100)

#Read metadata to get maps
with open("./data/metadata.txt", "r") as f:
    english_map, spanish_map, _, _ = f.read().split("\n")

english_map = loads(english_map)
spanish_map = loads(spanish_map)

#Find the reverse maps (id -> word)
english_reverse_map = {id: word for word, id in english_map.items()}
spanish_reverse_map = {id: word for word, id in spanish_map.items()}

english = data.english.to_list()
spanish = data.spanish.to_list()

for i in range(len(english)):
    english[i] = loads(english[i])
    spanish[i] = loads(spanish[i])

english = np.array(english)
spanish = np.array(spanish)

#Convert predicted probabilities to numerical sentences by
#Taking the ID with the highest probability
def map_predictions_to_numerical_sentences(preds):
    all_ids = []
    for i in range(len(preds)):
        ids = []
        for probs in preds[i]:
            id = np.argmax(probs)
            ids.append(id)
        all_ids.append(ids)
    return all_ids

#Convert numerical sentences to string sentences using reverse maps
#Ignore ID's of 0 as those were the length extension IDs
def map_numerical_sentence_to_strings(num_sentences, word_map):
    word_sentences = []
    for ids in num_sentences:
        sentence = []
        for id in ids:
            if id != 0:
                sentence.append(word_map[id])
        word_sentences.append(" ".join(sentence))
    return word_sentences

#Load the models
model1 = tf.keras.models.load_model('./models/lstm_full_256.h5')
model2 = tf.keras.models.load_model('./models/gru_full_256.h5')
model3 = tf.keras.models.load_model('./models/gru_full_512.h5')

#Make predictions with each model on the 100 samples
spanish_hat1 = model1.predict(english)
spanish_hat2 = model2.predict(english)
spanish_hat3 = model3.predict(english)

#Convert original english and spanish sentences back to words
eng_sentences = map_numerical_sentence_to_strings(english, english_reverse_map)
spa_sentences = map_numerical_sentence_to_strings(spanish, spanish_reverse_map)

#Get word sentences for all predictions
num_preds1 = map_predictions_to_numerical_sentences(spanish_hat1)
word_preds1 = map_numerical_sentence_to_strings(num_preds1, spanish_reverse_map)

num_preds2 = map_predictions_to_numerical_sentences(spanish_hat2)
word_preds2 = map_numerical_sentence_to_strings(num_preds2, spanish_reverse_map)

num_preds3 = map_predictions_to_numerical_sentences(spanish_hat3)
word_preds3 = map_numerical_sentence_to_strings(num_preds3, spanish_reverse_map)

#Store output of all in a new file
out_data = pd.DataFrame({"english_original": eng_sentences,
                         "spanish_original": spa_sentences,
                         "spanish_lstm": word_preds1,
                         "spanish_gru": word_preds2,
                         "spanish_gru_512": word_preds3})

out_data.to_csv("./predictions/sample_predictions_on_dataset.csv", sep="\t")

#Convert word sentences to numerical to send to models
#Make length 49 (expected by model)
def convert_numerical(language, word_map):
    result = []
    for sentence in language:
        sentence_split = sentence.split()
        for i in range(len(sentence_split)):
            sentence_split[i] = word_map[sentence_split[i]]
        missing = 49 - len(sentence_split)
        sentence_split.extend([0 for _ in range(missing)])
        result.append(sentence_split)
    return result

#Custom sentences to test (not from data set)
custom_sentences = ["be careful with that butter knife",
                    "she was the type of girl who wanted to live in a pink house",
                    "he always wore his sunglasses at night",
                    "he hated that he loved what she hated about hate",
                    "she had the gift of being able to paint songs",
                    "she works two jobs to make ends meet at least that was her reason for not having time to join us",
                    "they got there early and they got really good seats",
                    "he barked orders at his daughters but they just stared back with amusement",
                    "he looked behind the door and didnt like what he saw",
                    "while on the first date he accidentally hit his head on the beam",
                    "with a single flip of the coin his life changed forever",
                    "i really want to go to work but i am too sick to drive",
                    "he went back to the video to see what had been recorded and was shocked at what he saw",
                    "love is not like pizza",
                    "i am never at home on sundays"]

#Convert custom sentences to numerical
custom_numerical = np.array(convert_numerical(custom_sentences, english_map))

#Make predictions on custom sentences
spanish_hat1 = model1.predict(custom_numerical)
spanish_hat2 = model2.predict(custom_numerical)
spanish_hat3 = model3.predict(custom_numerical)

#Get word sentences for all predictions
num_preds1 = map_predictions_to_numerical_sentences(spanish_hat1)
word_preds1 = map_numerical_sentence_to_strings(num_preds1, spanish_reverse_map)

num_preds2 = map_predictions_to_numerical_sentences(spanish_hat2)
word_preds2 = map_numerical_sentence_to_strings(num_preds2, spanish_reverse_map)

num_preds3 = map_predictions_to_numerical_sentences(spanish_hat3)
word_preds3 = map_numerical_sentence_to_strings(num_preds3, spanish_reverse_map)

#Store output in file
out_data = pd.DataFrame({"english_original": custom_sentences,
                         "spanish_lstm": word_preds1,
                         "spanish_gru": word_preds2,
                         "spanish_gru_512": word_preds3})

out_data.to_csv("./predictions/predictions_on_custom.csv", sep="\t")
