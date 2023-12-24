import logging
import tensorflow as tf
from keras.layers.experimental.preprocessing import TextVectorization
import os
logging.basicConfig(filename='log.log', filemode='w', level=logging.DEBUG)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Vectorization:
    def __init__(self):
        self.binary_vectorize_layer = self.do_binary_vector_layer()

    @staticmethod
    def do_binary_vector_layer() -> TextVectorization:
        binary_vectorize_layer = TextVectorization(
            max_tokens=10000,
            output_mode='binary',
        )
        logging.debug('[LOG] MAKE BINARY LAYER')
        return binary_vectorize_layer

    @staticmethod
    def binary_vectorize_text(text):
        input_text = text[:-1]
        target_text = text[1:]
        return input_text, target_text