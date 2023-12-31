from vectorization_data import Vectorization
from tensorflow._api.v2.strings import unicode_split
from tensorflow.python.data import Dataset


class __Dataset:
    def __init__(self, path_to_file=''):
        text = open('data/habr_data_training/file1.txt', 'rb').read().decode(encoding='utf-8')
        self.layers = Vectorization(text)
        self.all_ids = self.layers.ids_from_chars_layer(unicode_split(text, 'UTF-8'))
        self.dataset_from_ids = Dataset.from_tensor_slices(self.all_ids)

        print(list(self.dataset_from_ids.as_numpy_iterator()))


if __name__ == '__main__':
    dataset = __Dataset()