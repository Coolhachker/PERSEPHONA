import logging
import tensorflow.python.data
import tensorflow as tf
from vectorization import Vectorization
from tensorflow.python.data import Dataset, TextLineDataset
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

    @property
    def set_dataset(self) -> Dataset:
        try:
            train_dataset = TextLineDataset([self.train_dir])
            logging.debug('[LOG] SET DATASET')
            return train_dataset
        except ValueError:
            pass

    def adapt_layer(self):
        logging.debug('[LOG] ADAPT LAYER')
        train_text = self.dataset.map(lambda text, labels: text)
        self.vectors.binary_vectorize_layer.adapt(train_text)

    def pre_treatment(self) -> Dataset:
        binary_train_dataset = self.dataset.map(self.vectors.binary_vectorize_text)
        return binary_train_dataset

    def configure_dataset(self) -> Dataset:
        logging.debug('[LOG] CONFIGURE DATASET')
        return self.binary_train_dataset.cache().prefetch(buffer_size=tf.python.data.AUTOTUNE)

