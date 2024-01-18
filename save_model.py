from tensorflow import saved_model
from generate_text import GenerateTextOneStepPERSEPHONA
from tensorflow import constant
import os
import logging
logging.basicConfig(filename='data/log.log', filemode='w', level=logging.DEBUG)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def save_model():
    generator = GenerateTextOneStepPERSEPHONA()
    state1 = None
    state2 = None
    next_char = constant(['build'])
    result = [next_char]

    for n in range(2):
        next_char, state1, state2 = generator.generate_text_one_step_model(next_char, state1=state1, state2=state2)
        result.append(next_char)

    saved_model.save(generator, "PERSEPHONA")


if __name__ == '__main__':
    save_model()