import os.path
from keras.callbacks import ModelCheckpoint
from set_dataset import DATASET
from set_model import PERSEPHONA
from keras.losses import SparseCategoricalCrossentropy


class Education:
    def __init__(self):
        self.dataset_obj = DATASET()
        self.dataset = self.dataset_obj.return_dataset()
        vocab_size = len(self.dataset_obj.layers.ids_from_chars_layer.get_vocabulary())
        embedding_dim = 256
        rnn_units = 1024

        self.model = PERSEPHONA(vocab_size, embedding_dim, rnn_units)
        self.compile_model()
        self.checkpoint_callback = self.create_checkpoints()

    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    @staticmethod
    def create_checkpoints():
        checkpoints_prefix = os.path.join('checkpoints', 'ckpt_{epoch}')
        return ModelCheckpoint(
            filepath=checkpoints_prefix,
            save_weights_only=True
        )


if __name__ == '__main__':
    education = Education()
