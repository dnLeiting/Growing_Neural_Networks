import unittest

import tensorflow as tf
from keras import initializers

from nng.grow.models import create_nng


class Testnng_MLP_MODEL(unittest.TestCase):
    def test_create_nng(self):
        """Testing the create_nng function

        """
        model = create_nng(model_type='mlp', out_dim=10)
        print(model)

        inp = tf.random.normal(shape=(1, 784))

        print(model(inp))

        print(model)


class Testnng_STUDENT_MODEL(unittest.TestCase):
    def test_create_student_nng(self):
        """Testing the create_student_nng function

        """
        w_init = initializers.RandomUniform(minval=-1, maxval=1)
        b_init = initializers.RandomUniform(minval=-1, maxval=1)

        model = create_nng(model_type='student_mlp', m_i=20, m_h=[10], m_o=10,
                           kernel_init=w_init, bias_init=b_init)
        print(model)

        model2 = create_nng(model_type='student_mlp',
                            m_i=20, m_h=[10, 5, 4], m_o=10,
                            kernel_init=w_init, bias_init=b_init)
        print(model2)

        inp = tf.random.normal(shape=(30, 20))
        print(model(inp))
        print(model2(inp))

        print(model)
        print(model2)


if __name__ == "__main__":
    unittest.main()
