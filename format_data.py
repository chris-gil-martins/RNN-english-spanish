'''
Format cleaned data so it can be used for the neural network
'''

import pandas as pd
from json import dumps

#Given all the sentences/phrases in the language data,
#save all unique words into the vocabulary
def get_vocabulary(language):
    vocab = []
    for i in range(len(language)):
        phrase = language[i]
        words = phrase.split()
        for word in words:
            if word not in vocab:
                vocab.append(word)
    return vocab

#Convert string sentences into numerical sentences by using a vocabulary map
def convert_numerical(language, word_map):
    result = []
    for sentence in language:
        sentence_split = sentence.split()
        for i in range(len(sentence_split)):
            sentence_split[i] = word_map[sentence_split[i]]
        result.append(sentence_split)
    return result

#Extend the length of numerical sentences to be equal to the longest sentence
#in the data so dimensions match for the neural network. Extend by appending
#0's to the end of the sentence.
def extend_sentence(language_numerical, longest_sentence):
    result = []
    for i in range(len(language_numerical)):
        sentence = language_numerical[i]
        missing = longest_sentence - len(sentence)
        sentence.extend([0 for _ in range(missing)])
        result.append(sentence)
    return result


#Read data
data = pd.read_csv("./data/cleaned_data.csv", sep="\t")
english = data.english.to_list()
spanish = data.spanish.to_list()

#Get the vocabulary for each language
english_vocab = get_vocabulary(english)
spanish_vocab = get_vocabulary(spanish)

#Create a numerical map for each language
english_map = {english_vocab[i]: i+1 for i in range(len(english_vocab))}
spanish_map = {spanish_vocab[i]: i+1 for i in range(len(spanish_vocab))}

#Convert all string sentences into numerical sentences
english_numerical = convert_numerical(english, english_map)
spanish_numerical = convert_numerical(spanish, spanish_map)

#Get the length of the longest sentence in each language
english_longest = len(max(english_numerical,  key=lambda s : len(s)))
spanish_longest = len(max(spanish_numerical,  key=lambda s : len(s)))

#Determine overall longest sentence
total_longest = english_longest
if spanish_longest > total_longest:
    total_longest = spanish_longest

#Extend all sentences to be equal length
english_formatted = extend_sentence(english_numerical, total_longest)
spanish_formatted = extend_sentence(spanish_numerical, total_longest)

#Update and save data
data.english = english_formatted
data.spanish = spanish_formatted
data.to_csv("./data/formatted_data.csv", index=False, sep="\t")

#Store maps and vocabulary counts to be easily accessed later
with open("./data/metadata.txt", "w") as f:
    f.write(dumps(english_map) + "\n")
    f.write(dumps(spanish_map) + "\n")
    f.write(str(len(english_vocab)) + "\n")
    f.write(str(len(spanish_vocab)))
