from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from SPECtogram import gimmeDaSPECtogram
import numpy as np
from keras.utils import to_categorical
import os

DATA_PATH = "./data/"
feature_dim_1 = 97
feature_dim_2 = 12
channel = 1


json_file = open('model_3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_3.h5")
print("Loaded model from disk")

# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    #labels = os.listdir(path)
    labels = ['up', 'down', 'left', 'right', 'one', 'two', 'three', 'four', 'stop', 'go']
    #print(labels)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


# Predicts one sample
def predict(filepath, model):
    sample = gimmeDaSPECtogram(filepath)
    print(sample.shape)
    while sample.shape[1] > 97:
        sample = sample[:,:-1].copy()
        #print(sample.shape)

    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]

print(predict("data/down/0ba018fc_nohash_2.wav", loaded_model))
print(predict("data/go/37dca74f_nohash_1.wav", loaded_model))
print(predict("MeFour.wav", loaded_model))