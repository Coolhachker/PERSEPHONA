from keras.layers import StringLookup
from tensorflow._api.v2.strings import reduce_join
import logging


class Vectorization:
    def __init__(self, path):
        self.path = path
        self.ids_from_chars_layer = self.ids_from_chars()
        self.chars_from_ids_layer = self.chars_from_ids()

    def ids_from_chars(self):
        """
        Функция преобразует обычные слова в векторы данных

        :return: Возвращает слой, в котором находятся все векторы символов
        """

        with open(self.path, 'r') as data:
            vocab = []
            count: int = 0
            for string in data:
                if count < 50000:
                    vocab.extend(sorted(set(string)))
                    vocab = list(sorted(set(vocab)))
                    count += 1
                    if count % 100 == 0:
                        logging.info(f'[LAYER] Count is {count}/50000')
                else:
                    break

        return StringLookup(
            vocabulary=list(sorted(set(vocab))),
            mask_token=None
        )

    def chars_from_ids(self):
        """
        Функция преобразует векторы данных в обычные слова

        :return: Возвращает слой, в котором находятся все символы
        """
        return StringLookup(
            vocabulary=self.ids_from_chars_layer.get_vocabulary(),
            invert=True,
            mask_token=None
        )

    def text_from_ids(self, ids):
        """
        Функция объединяет элементы тензора в единый элемент

        :param ids: векторы данных
        :return:
        """
        return reduce_join(self.chars_from_ids_layer(ids), axis=-1)