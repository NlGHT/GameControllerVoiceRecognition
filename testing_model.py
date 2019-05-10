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


json_file = open('model_4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_4.h5")
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
print(predict("MeDown.wav", loaded_model))
print(predict("data/down/0c540988_nohash_0.wav", loaded_model))
print(predict("data/two/0c2d2ffa_nohash_0.wav", loaded_model))
print(predict("MeTwo.wav", loaded_model))
print(predict("data/two/0c2d2ffa_nohash_0.wav", loaded_model))
print(predict("data/four/0ac15fe9_nohash_0.wav", loaded_model))
print(predict("MeFour.wav", loaded_model))
print(predict("data/four/0bfec55f_nohash_0.wav", loaded_model))
print(predict("data/right/0d393936_nohash_0.wav", loaded_model))
print(predict("MeRight.wav", loaded_model))
print(predict("data/right/0d53e045_nohash_0.wav", loaded_model))
print(predict("data/left/0d82fd99_nohash_2.wav", loaded_model))
print(predict("MeLeft.wav", loaded_model))
print(predict("data/left/0d6d7360_nohash_2.wav", loaded_model))