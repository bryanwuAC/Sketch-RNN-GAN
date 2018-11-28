import numpy as np
import torch
from hyperparameters import HyperParameters
from torch.autograd import Variable


class DataLoader:
    def __init__(self, data):
        self.hp = HyperParameters()
        self.data = data
        self.clean_data()
        self.normalize()
        self.N_max = self.get_max_length()

    def get_max_length(self):
        length_list = [len(sample) for sample in self.data]
        return max(length_list)

    def clean_data(self):
        cleaned_data = []
        for sample in self.data:
            sample = np.minimum(sample, self.hp.limit)
            sample = np.maximum(sample, -self.hp.limit)
            sample = np.array(sample, dtype=np.float32)
            cleaned_data.append(sample)
        self.data = cleaned_data

    def get_normalize_scale_factor(self):
        temp_data = []
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                temp_data.append(self.data[i][j][0])
                temp_data.append(self.data[i][j][1])
        return np.std(np.array(temp_data))

    def normalize(self):
        self.scale_factor = self.get_normalize_scale_factor()
        for i in range(len(self.data)):
            self.data[i][:, 0:2] /= self.scale_factor

    def get_batch(self, batch_size):
        indices = np.random.choice(len(self.data), batch_size)
        sequences = [self.data[index] for index in indices]
        transformed_sequences = []
        sequence_lengths = []
        for sequence in sequences:
            sequence_length = sequence.shape[0]
            sequence_lengths.append(sequence_length)

            # Initialize new 5 stroke sequence
            transformed_sequence = np.zeros((self.N_max, 5))
            # Copy pen location
            transformed_sequence[:sequence_length, :2] = sequence[:, :2]
            # Copy pen state of every element in sequence
            transformed_sequence[:sequence_length, 2] = 1 - sequence[:, 2]
            transformed_sequence[:sequence_length, 3] = sequence[:, 2]
            # Mark pen state of EOS for Si >= sequence_length
            transformed_sequence[sequence_length:, 2] = 0
            transformed_sequence[sequence_length:, 3] = 0
            transformed_sequence[sequence_length:, 4] = 1
            # Append pen state of SOS at S0
            transformed_sequence = np.vstack((np.array([0, 0, 1, 0, 0]), transformed_sequence))
            transformed_sequences.append(transformed_sequence)
        transformed_sequences = Variable(torch.from_numpy(np.stack(transformed_sequences, 1)).cuda().float())
        return transformed_sequences, sequence_lengths
