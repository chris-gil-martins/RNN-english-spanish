'''
Plot training accuracies per epoch for all models
'''

import matplotlib.pyplot as plt
import pandas as pd

lstm_reduced_logs = pd.read_csv("./logs/log_lstm_reduced_512.csv", sep=";")
plt.title("Accuracy of LSTM Model on Reduced Data")
plt.plot(lstm_reduced_logs.epoch, lstm_reduced_logs.accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()

gru_reduced_logs = pd.read_csv("./logs/log_gru_reduced_512.csv", sep=";")
plt.title("Accuracy of GRU Model on Reduced Data")
plt.plot(gru_reduced_logs.epoch, gru_reduced_logs.accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()

plt.title("Accuracy Comparison on Reduced Data")
plt.plot(lstm_reduced_logs.epoch, lstm_reduced_logs.accuracy)
plt.plot(gru_reduced_logs.epoch, gru_reduced_logs.accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["LSTM", "GRU"])
plt.show()

lstm_full_logs = pd.read_csv("./logs/log_lstm_full_256.csv", sep=";")
plt.title("Accuracy of LSTM Model on Full Data")
plt.plot(lstm_full_logs.epoch, lstm_full_logs.accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()

gru_full_256_logs = pd.read_csv("./logs/log_gru_full_256.csv", sep=";")
plt.title("Accuracy of GRU Model on Full Data")
plt.plot(gru_full_256_logs.epoch, gru_full_256_logs.accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()

gru_full_512_logs = pd.read_csv("./logs/log_gru_full_512.csv", sep=";")
plt.title("Accuracy of GRU Model on Full Data (512 batch)")
plt.plot(gru_full_512_logs.epoch, gru_full_512_logs.accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()

plt.title("Accuracy Comparison on Full Data")
plt.plot(lstm_full_logs.epoch, lstm_full_logs.accuracy)
plt.plot(gru_full_256_logs.epoch, gru_full_256_logs.accuracy)
plt.plot(gru_full_512_logs.epoch, gru_full_512_logs.accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["LSTM", "GRU", "GRU (512)"])
plt.show()
