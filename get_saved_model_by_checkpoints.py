from set_model import PERSEPHONA
from vectorization_data import Vectorization
from tensorflow._api.v2.strings import unicode_split
from tensorflow import constant
from tensorflow import train
from tensorflow import saved_model
from keras.models import save_model


def get_saved_model():
    layers = Vectorization('data/habr_data_training/habr_DEVELOP.txt', 2000000)
    vocab_size = len(layers.ids_from_chars_layer.get_vocabulary())
    persephona = PERSEPHONA(vocab_size, 256, 1024)

    persephona.compile()

    first_promt = constant(['input'])
    __input__ = layers.ids_from_chars_layer(unicode_split(first_promt, 'UTF-8'))

    latest_checkpoint = train.latest_checkpoint('checkpoints')

    persephona.load_weights(latest_checkpoint)

    return persephona


if __name__ == '__main__':
    get_saved_model()
