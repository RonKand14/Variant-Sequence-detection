from Bio import SeqIO
import pandas as pd
import numpy as np
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.layers import BatchNormalization
from keras import initializers


class Helpers:
    
    def __init__(self, csv_path, fasta_path):
        self.csv_path = csv_path
        self.fasta_path = fasta_path
        self.linages = None
        self.mapping_dict = dict()
        self.genome_to_linage = dict()
        self.names = None
        self.seq = None
        self.model = None
        self.labels = []
        self.epochs = None

    def parseFasta(self):
        seq_list = []
        name_list = []
        fasta_sequences = SeqIO.parse(open(self.fasta_path), 'fasta')
        index = 0
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            print(f"index {index}")
            index = index+1
            splited_name = name.split("|")[1]
            name_list.append(splited_name)
            seq_list.append(sequence)
            if index == 10000:
                break
        print("Done fasta ")
        self.names = name_list
        self.seq = seq_list
        return name_list, seq_list

    def parseAccToLinage(self):
        df = pd.read_csv(self.csv_path)
        acc_id = df["accession_id"]
        linage = df["lineage"]
        linage.drop_duplicates(inplace=True)
        self.linages = linage
        for value in linage:
            self.mapping_dict[value] = []
            self.genome_to_linage[value] = []
        for row in df.iterrows():
            id = row[1]["accession_id"]
            lin = row[1]["lineage"]
            self.mapping_dict[lin].append(id)

    def mapGenomeToLinage(self):
        for index in range(0, len(self.names)):
            name = self.names[index]
            genome = self.seq[index]
            for key in self.mapping_dict.keys():
                if name in self.mapping_dict[key]:
                    self.genome_to_linage[key].append(genome)

    def oneHot(self, seq):
        seq2 = list()
        #mapping = {"A": [1., 0., 0., 0.], "C": [0., 1., 0., 0.], "G": [0., 0., 1., 0.], "T": [0., 0., 0., 1.]}
        mapping = {"A": 2, "C": 3, "G": 4, "T": 5}
        for i in seq:
            seq2.append(mapping[i] if i in mapping.keys() else 1)
        return np.array(seq2)

    def mapGenomeToVector(self):
        features = []
        for se in self.seq:
            features.append(self.oneHot(se))
            self.getGenomeKey(se)
        return features

    def getGenomeKey(self, seq):
        for key in self.genome_to_linage:
            if seq in self.genome_to_linage[key]:
                self.labels.append(key)
                break

    def initModel(self, train_features, n_output):
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=7, input_shape=(train_features.shape[1], 1),
                         padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.5))
        model.add(Conv1D(filters=24, kernel_size=12, input_shape=(train_features.shape[1], 1),
                         padding='same', activation='relu'))
        # model.add(MaxPooling1D(pool_size=2))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Flatten())
        #model.add(Dense(n_output, activation='sigmoid', kernel_initializer=initializers.RandomNormal(stddev=0.01),bias_initializer=initializers.Zeros()))
        # model.add(Dropout(0.2))
        #model.add(Dense(n_output, activation='softmax'))
        model.add(Dense(n_output, activation='sigmoid', kernel_initializer=initializers.RandomNormal(stddev=0.01),bias_initializer=initializers.Zeros()))
        lrate = 0.1
        # decay = lrate / epochs
        decay = 0.0025
        sgd = SGD(learning_rate=lrate, momentum=0.90, decay=decay, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        model.summary()
        self.model = model
