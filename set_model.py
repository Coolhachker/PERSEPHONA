from keras.models import Model
from keras.layers import Embedding, LSTM, Dense


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

    def call(self, inputs, states=None, return_state=False, training=False):
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