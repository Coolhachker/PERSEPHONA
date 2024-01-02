from keras.models import Model
from train_model import Education
from tensorflow import function
from tensorflow._api.v2.strings import unicode_split
from tensorflow._api.v2.random import categorical
from tensorflow import squeeze, constant


class OneStepPERSEPHONA(Model):
    def __init__(self):
        super().__init__()
        self.temperature = 1.0
        self.education = Education()
        self.model = self.education.model
        self.ids_from_chars_layer = self.education.dataset_obj.layers.ids_from_chars_layer

    @function
    def generate_text_one_step_model(self, inputs, states=None):
        input_ids = self.ids_from_chars_layer(unicode_split(inputs, 'UTF-8')).to_tensor()

        predicted_logits, states = self.model(inputs=input_ids, states=states, return_states=True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature

        predicted_ids = categorical(predicted_logits, num_samples=1)
        predicted_ids = squeeze(predicted_ids, axis=-1)

        predicted_chars = self.education.dataset_obj.layers.text_from_ids(predicted_ids)

        return predicted_chars, states


if __name__ == '__main__':
    generator = OneStepPERSEPHONA()
    states = None
    next_char = constant(['test'])
    result = [next_char]

    for n in range(10):
        pass




