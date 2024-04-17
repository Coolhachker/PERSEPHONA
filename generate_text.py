from keras.models import Model
from train_model import Education
from tensorflow import function
from tensorflow._api.v2.strings import unicode_split, join
from tensorflow._api.v2.random import categorical
from tensorflow import squeeze, constant
import tensorflow as tf
from tensorflow import saved_model, TensorSpec, float32, string, Tensor
from tensorflow import saved_model
from vectorization_data import Vectorization


class GenerateTextOneStepPERSEPHONA(Model):
    def __init__(self):
        super().__init__()
        self.temperature = 1.0
        # self.education = Education()
        vectorization_layers = Vectorization('data/habr_data_training/habr_DEVELOP.txt', 2000000)
        self.model = saved_model.load('PERSEPHONA_R')
        self.ids_from_chars_layer = vectorization_layers.ids_from_chars_layer
        self.chars_from_ids_layer = vectorization_layers.chars_from_ids_layer

        skip_ids = self.ids_from_chars_layer(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            values=[-float('inf')] * len(skip_ids),
            indices=skip_ids,
            dense_shape=[len(self.ids_from_chars_layer.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @function
    def generate_text_one_step_model(self, inputs, state1=None, state2=None):
        state = [state1, state2] if state1 is not None and state2 is not None else None
        input_ids = self.ids_from_chars_layer(unicode_split(inputs, 'UTF-8'))

        predicted_logits, states_cell = self.model(inputs=input_ids, states=state, return_state=True)
        predicted_logits = predicted_logits.to_tensor()
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = categorical(predicted_logits, num_samples=1)
        predicted_ids = squeeze(predicted_ids, axis=-1)

        predicted_chars = self.chars_from_ids_layer(predicted_ids)

        return predicted_chars, states_cell[0], states_cell[1]


def generate_text(__input__: str):
    persephona = saved_model.load("PERSEPHONA")
    state1 = None
    state2 = None
    next_char = constant([__input__])
    result = [next_char]

    for n in range(100):
        next_char, state1, state2 = persephona.generate_text_one_step_model(next_char, state1=state1, state2=state2)
        result.append(next_char)

    return join(result)[0].numpy().decode('utf-8')


if __name__ == '__main__':
    print(generate_text('программирование'))






