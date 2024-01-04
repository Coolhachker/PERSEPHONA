from tensorflow import saved_model
from generate_text import GenerateTextOneStepPERSEPHONA


def save_model():
    generator = GenerateTextOneStepPERSEPHONA()
    saved_model.save(generator, 'PERSEPHONA')


if __name__ == '__main__':
    save_model()