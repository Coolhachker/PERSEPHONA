from keras.models import Model
from keras.layers import Embedding, LSTM, Dense
from tensorflow import GradientTape, argmax
from keras.metrics import Accuracy
import numpy


class PERSEPHONA(Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = Embedding(
            vocab_size,
            embedding_dim
        )
        self.lstm = LSTM(
            rnn_units,
            return_sequences=True,
            return_state=True
        )
        self.dense = Dense(
            vocab_size
        )
        self.metric = Accuracy()

    def call(self, inputs, states=None, return_state=False, training=False):
        """
        Когда идет обращение к экземпляру класса, то идет обращение к этой функции при помощи перегрузки оператор __call__(...)

        :param inputs: Тензор данных
        :param states: состояние на прошлом шаге. [state1, stat2]
        :param return_state:
        :param training:
        :return:
        """
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.lstm.get_initial_state(x)
        x, final_memory_state, final_carry_state = self.lstm(x, initial_state=states, training=training)
        x = self.dense(x, training=training)
        if return_state:
            return x, [final_memory_state, final_carry_state]
        else:
            return x

    def train_step(self, data):
        """
        Функция для кастомного обучения. В этом случае обучение на градиенте ошибок.

        :param data:
        :return:
        """
        inputs, target = data

        with GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.loss(target, predictions)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {'loss': loss}