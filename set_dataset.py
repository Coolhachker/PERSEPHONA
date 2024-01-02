from vectorization_data import Vectorization
from tensorflow._api.v2.strings import unicode_split
from tensorflow.python.data import Dataset
from tensorflow.python.data.experimental import AUTOTUNE


class DATASET:
    def __init__(self, path_to_file=''):
        self.seq_length = 5
        self.batch_size = 1
        self.buffer_size = 10000

        self.text = open('data/habr_data_training/file1.txt', 'rb').read().decode(encoding='utf-8')

        self.layers = Vectorization(self.text)
        self.all_ids = self.set_ids()

        self.dataset_from_ids = self.set_dataset()
        self.sequences = self.set_sequences()

    def set_ids(self):
        return self.layers.ids_from_chars_layer(unicode_split(self.text, 'UTF-8'))

    def set_dataset(self):
        return Dataset.from_tensor_slices(self.all_ids)

    def set_sequences(self):
        return self.dataset_from_ids.batch(self.seq_length+1, drop_remainder=False)

    @staticmethod
    def split_input_target_data(sequences):
        input_text = sequences[:-1]
        target_text = sequences[1:]

        return input_text, target_text

    def set_packets_for_train(self, dataset_: Dataset):
        return dataset_.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    def return_dataset(self):
        return self.set_packets_for_train(self.sequences.map(self.split_input_target_data))