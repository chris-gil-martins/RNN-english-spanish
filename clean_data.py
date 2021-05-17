'''
Clean the raw English-Spanish data and save it to its own file
'''

import pandas as pd
import numpy as np
import re

#Read data and drop unnecessary metadata
data = pd.read_csv("./data/raw_data.txt", sep="\t", names=["english", "spanish", "meta"])
data.drop("meta", inplace=True, axis=1)

english = data.english.to_list()
spanish = data.spanish.to_list()

#Remove all punctuation from every sentence and make all words lowercase
for i in range(len(english)):
    s = re.sub(r'[^\w\s]','',english[i])
    english[i] = s.lower()

    s2 = re.sub(r'[^\w\s]','',spanish[i])
    spanish[i] = s2.lower()

#Code used to create reduced data set for testing purposes
# indices = []
# for i in range(len(english)):
#     word_split = english[i].split()
#     if len(word_split) > 10 and len(word_split) < 20:
#         indices.append(i)
# english = np.array(english)[indices]
# spanish = np.array(spanish)[indices]

english = np.array(english)
spanish = np.array(spanish)

data = pd.DataFrame({"english": english, "spanish": spanish})

#Drop duplicates in both languages and adjust indices
data.drop_duplicates(subset ="english",
                     keep = "first", inplace = True)
data = data.reset_index()
data.drop_duplicates(subset ="spanish",
                     keep = "first", inplace = True)
data = data.reset_index()

#Remove extra indexing columns
data.drop(["level_0", "index"], inplace=True, axis=1)

#Store cleaned data separately
data.to_csv("./data/cleaned_data.csv", index=False, sep="\t")
