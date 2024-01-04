from tensorflow import saved_model
from generate_text import GenerateTextOneStepPERSEPHONA


def save_model():
    generator = GenerateTextOneStepPERSEPHONA()
    saved_model.save(generator, 'PERSEPHONA')