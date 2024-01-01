from set_dataset import DATASET
from set_model import PERSEPHONA


class Education:
    def __init__(self):
        self.dataset_obj = DATASET()
        self.dataset = self.dataset_obj.return_dataset()

        vocab_size = len(self.dataset_obj.layers.ids_from_chars_layer.get_vocabulary())
        embedding_dim = 256
        rnn_units = 1024

        self.model = PERSEPHONA(vocab_size, embedding_dim, rnn_units)

    def test_model(self):
        pass