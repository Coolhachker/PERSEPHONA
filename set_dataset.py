import logging
import tensorflow.python.data
import tensorflow as tf
from vectorization import Vectorization
from tensorflow.python.data import Dataset, TFRecordDataset, TextLineDataset, AUTOTUNE
import os
logging.basicConfig(filename='log.log', filemode='w', level=logging.DEBUG)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Setter:
    def __init__(self, path):
        self.train_dir = path
        self.vectors = Vectorization()
        self.dataset = self.set_dataset
        self.adapt_layer()
        self.binary_train_dataset = self.pre_treatment()
        self.binary_train_dataset = self.configure_dataset()
        logging.debug(f'configure_dataset={self.binary_train_dataset}')

    @property
    def set_dataset(self) -> list:
        try:
            train_text = open(self.train_dir, 'r').readlines()
            logging.debug(f'set_dataset()={train_text}')
            logging.debug('[LOG] SET DATASET')
            return train_text
        except ValueError:
            pass

    def adapt_layer(self):
        logging.debug('[LOG] ADAPT LAYER')
        self.vectors.binary_vectorize_layer.adapt(self.dataset)

    def pre_treatment(self) -> Dataset:
        all_ids = self.vectors.binary_vectorize_layer(self.dataset)
        print(all_ids)
        self.dataset = Dataset.from_tensor_slices(all_ids)
        print(self.dataset)
        seq = self.dataset.batch(100, drop_remainder=True)
        print(seq)
        binary_train_dataset = seq.map(self.vectors.binary_vectorize_text)
        print(binary_train_dataset)
        logging.debug(f'pre_treatment={binary_train_dataset}')
        return binary_train_dataset

    def configure_dataset(self) -> Dataset:
        logging.debug('[LOG] CONFIGURE DATASET')
        return self.binary_train_dataset.shuffle(10000).batch(32, drop_remainder=True).prefetch(AUTOTUNE)

