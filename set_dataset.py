from vectorization_data import Vectorization
from tensorflow._api.v2.strings import unicode_split
from tensorflow.python.data import Dataset


class __Dataset:
    def __init__(self, path_to_file=''):
        self.seq_length = 5
        self.text = open('data/habr_data_training/file1.txt', 'rb').read().decode(encoding='utf-8')

        self.layers = Vectorization(self.text)
        self.all_ids = self.set_ids()

        self.dataset_from_ids = self.set_dataset()
        self.sequences = self.set_sequences()

        self.dataset = self.sequences.map(self.split_input_target_data)

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


if __name__ == '__main__':
    dataset = __Dataset()