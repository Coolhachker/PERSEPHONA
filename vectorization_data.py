from keras.layers import StringLookup
from tensorflow._api.v2.strings import reduce_join


class Vectorization:
    def __init__(self, data):
        self.vocab = sorted(set(data))
        self.ids_from_chars_layer = self.ids_from_chars()
        self.chars_from_ids_layer = self.chars_from_ids()

    def ids_from_chars(self):
        return StringLookup(
            vocabulary=list(self.vocab),
            mask_token=None
        )

    def chars_from_ids(self):
        return StringLookup(
            vocabulary=self.ids_from_chars_layer.get_vocabulary(),
            invert=True,
            mask_token=None
        )

    def text_from_ids(self, ids):
        return reduce_join(self.chars_from_ids_layer(ids), axis=-1)