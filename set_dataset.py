from vectorization_data import Vectorization
from tensorflow._api.v2.strings import unicode_split
from tensorflow.python.data import Dataset
from tensorflow.python.data.experimental import AUTOTUNE
from tensorflow import constant, concat
import logging
logging.basicConfig(filename='data/log.log', filemode='w', level=logging.DEBUG)


class DATASET:
    def __init__(self, path_to_file='data/habr_data_training/file1.txt'):
        self.seq_length = 40
        self.batch_size = 64
        self.buffer_size = 10000

        self.path = path_to_file

        self.layers = Vectorization(path_to_file)
        self.all_ids = constant([0], dtype='int64')
        self.set_ids()

        self.dataset_from_ids = self.set_dataset()
        self.sequences = self.set_sequences()

    def set_ids(self):
        """
        Функция векторизирует str символы в id значение

        :return:
        """
        with open(self.path, 'r') as data:
            count: int = 0
            for string in data:
                if count < 10000:
                    self.all_ids = concat([self.all_ids, self.layers.ids_from_chars_layer(unicode_split(string, 'UTF-8'))], axis=0)
                    count += 1
                    if count % 100 == 0:
                        logging.info(f'[DATASET] Count is {count}/10000')
                else:
                    break

    def set_dataset(self):
        """
        Функция создает Dataset из тензора векторизированных данных

        :return:
        """
        return Dataset.from_tensor_slices(self.all_ids)

    def set_sequences(self):
        """
        Функция создает пакеты данных из Dataset

        :return:
        """
        return self.dataset_from_ids.batch(self.seq_length+1, drop_remainder=True)

    @staticmethod
    def split_input_target_data(sequences):
        """
        Функция создает input и target значения внутри Dataset
        Пример: split_input_target_data('Tensorflow') -> [['e', 'n', 's', 'o', 'r', 'f', 'l', 'o', 'w'], ['T', 'e', 'n', 's', 'o', 'r', 'f', 'l', 'o']]

        :param sequences:
        :return:
        """
        input_text = sequences[:-1]
        target_text = sequences[1:]

        return input_text, target_text

    def set_packets_for_train(self, dataset_: Dataset):
        """
        Создает обучающие пакеты данных из Dataset

        :param dataset_:
        :return:
        """
        return dataset_.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True).prefetch(AUTOTUNE)

    def return_dataset(self):
        """
        Функция возвращает Dataset

        :return:
        """
        return self.set_packets_for_train(self.sequences.map(self.split_input_target_data))