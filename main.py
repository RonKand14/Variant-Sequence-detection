from Helper import Helpers
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np
# from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import pandas as pd
import csv


# Data Path
print("Step 1: Data path init")
input_file = "./data/final/ena_data_20230624-2230.fasta"
csv_path = "./data/final/covid19data_embl-covid19.tsv.csv"
output_file = "./data/output.txt"
load = True     # load from file
continue_fit = True
# Helper class Init
print("Helper class init")
helper = Helpers(csv_path, input_file)

# Data Parsing (ID to Lineage + Genome to Lineage) - All values will be stores in dicts
print("Data Parsing (ID to Lineage + Genome to Lineage) - All values will be stores in dicts")
if not load:
    helper.parseAccToLinage()
    names, seq = helper.parseFasta()
    helper.mapGenomeToLinage()

    # Data decoder (Genome as Vectors instead of strings)
    print("Data decoder (Genome as Vectors instead of strings)")
    gen_features = helper.mapGenomeToVector()
    label = helper.labels
    labels = label
    for i in range(0,19):
        label = np.append(label, labels[0])
    padding = tf.keras.preprocessing.sequence.pad_sequences(gen_features, padding="post")
    input_features = np.stack(padding)

    one_hot_encoder = OneHotEncoder()
    label = np.array(label).reshape(-1, 1)
    input_labels = one_hot_encoder.fit_transform(label).toarray()
else:
    input_features = np.memmap('data.bin', mode='r+', dtype=int, shape=(25000, 31190))
    input_features = np.array(input_features)
    input_labels = np.memmap('label_2.bin', mode='r+', dtype=int, shape=(25000, 369))
    input_labels = np.array(input_labels)

# Data Split (Train, Validation and test)
train_features, test_features, train_labels, test_labels = train_test_split(
    input_features, input_labels, test_size=0.20, random_state=42)

# Model Init
print("Model")
n_output = input_labels.shape[1]
np.expand_dims(train_features, 0)
batch_size = 24
val_split = 0.3
epochs = 1

# Model Training
filepath_read = "./model/model_1.h5"
if continue_fit:
    # fit the new model
    new_model = load_model(filepath_read)
    checkpoint = ModelCheckpoint(filepath_read, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    new_model.fit(train_features, train_labels, verbose=1, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=val_split)
    helper.model = new_model
else:
    helper.initModel(train_features, n_output)

# Model Training
filepath = "./model/model_2.h5"
# checkpoint
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Model testing
x = helper.model.evaluate(test_features, test_labels)
predicted_labels = helper.model.predict(np.stack(test_features))
scores = helper.model.evaluate(test_features, test_labels, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
