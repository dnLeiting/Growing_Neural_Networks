import tensorflow as tf
from tensorflow import keras
from keras import initializers
from keras.layers import Dense


class TS_Model(keras.Model):
    def __init__(self, m_i, m_h, m_o, seed=42, special_init=True):
        super().__init__()
        tf.random.set_seed(seed)
        if special_init:
            init = initializers.RandomUniform(minval=-1, maxval=1)
            kwargs = {'kernel_initializer': init,
                      'bias_initializer': init}
        else:
            kwargs = {}

        self.m_i = m_i
        self.h_lay = Dense(m_h,
                           activation='relu',
                           **kwargs)
        self.out_lay = Dense(m_o,
                             **kwargs)

    def call(self, x):
        x = self.h_lay(x)
        return self.out_lay(x)
