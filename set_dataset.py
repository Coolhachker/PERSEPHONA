from vectorization_data import Vectorization
from tensorflow._api.v2.strings import unicode_split
from tensorflow.python.data import Dataset


class __Dataset:
    def __init__(self, path_to_file=''):
        self.seq_length = 100
        self.text = open('data/habr_data_training/file1.txt', 'rb').read().decode(encoding='utf-8')
        self.layers = Vectorization(self.text)
        self.all_ids = self.set_ids()
        self.dataset_from_ids = self.set_dataset()

    def set_ids(self):
        return self.layers.ids_from_chars_layer(unicode_split(self.text, 'UTF-8'))

    def set_dataset(self):
        return Dataset.from_tensor_slices(self.all_ids)

    def set_sequences(self):
        pass


if __name__ == '__main__':
    dataset = __Dataset()