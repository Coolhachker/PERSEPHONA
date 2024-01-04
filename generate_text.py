from keras.models import Model
from train_model import Education
from tensorflow import function
from tensorflow._api.v2.strings import unicode_split, join
from tensorflow._api.v2.random import categorical
from tensorflow import squeeze, constant
import tensorflow as tf
from tensorflow import saved_model


class GenerateTextOneStepPERSEPHONA(Model):
    def __init__(self):
        super().__init__()
        self.temperature = 1.0
        self.education = Education()
        self.model = self.education.model
        self.ids_from_chars_layer = self.education.dataset_obj.layers.ids_from_chars_layer
        self.chars_from_ids_layer = self.education.dataset_obj.layers.chars_from_ids_layer

        skip_ids = self.ids_from_chars_layer(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            values=[-float('inf')] * len(skip_ids),
            indices=skip_ids,
            dense_shape=[len(self.ids_from_chars_layer.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @function
    def generate_text_one_step_model(self, inputs, state=None):
        input_ids = self.ids_from_chars_layer(unicode_split(inputs, 'UTF-8'))

        predicted_logits, states_cell = self.model(inputs=input_ids, states=state, return_state=True)
        predicted_logits = predicted_logits.to_tensor()
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = categorical(predicted_logits, num_samples=1)
        predicted_ids = squeeze(predicted_ids, axis=-1)

        predicted_chars = self.chars_from_ids_layer(predicted_ids)

        return predicted_chars, states_cell


def generate_text(__input__: str):
    persephona = saved_model.load("PERSEPHONA")
    states = None
    next_char = constant([__input__])
    result = [next_char]

    for n in range(250):
        next_char, states = persephona.generate_text_one_step_model(next_char, state=states)
        result.append(next_char)

    return join(result)[0].numpy().decode('utf-8')





