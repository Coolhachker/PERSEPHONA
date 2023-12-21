from set_dataset import Setter
from keras.layers import Dense, Dropout, LSTM
from keras import Model
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import ModelCheckpoint
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(filename='log.log', filemode='w', level=logging.DEBUG)


class BinaryModel:
    def __init__(self):
        self.setter_dataset = Setter('data')
        self.binary_model = self.set_binary_model()
        self.vocabulary = self.setter_dataset.vectors.binary_vectorize_layer.get_vocabulary()
        self.compile_binary_model()
        self.checkpoint_callback = self.check_points()
        self.fit_model()

    def set_binary_model(self) -> Model:
        model = Model(([
            Dropout(0.5),
            LSTM(1024),
            Dense(self.vocabulary)
        ]))
        logging.debug('[LOG] SET BINARY MODEL')
        return model

    def compile_binary_model(self):
        self.binary_model.compile(
            loss=SparseCategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['accuracy']
        )
        logging.debug('[LOG] COMPILE MODEL')

    @staticmethod
    def check_points():
        checkpoint_prefix = os.path.join('checkpoints', "ckpt_{epoch}")
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True
        )
        logging.debug('[LOG] CREATE CHECKPOINTS')
        return checkpoint_callback

    def fit_model(self):
        logging.debug('[LOG] FIT MODEL')
        self.binary_model.fit(self.setter_dataset.binary_train_dataset, epochs=100, callbacks=[self.checkpoint_callback])


if __name__ == '__main__':
    binary_model = BinaryModel()
