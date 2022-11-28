import unittest

import numpy as np
from keras.layers import Dense
from tensorflow import keras

from nng.grow.layers import nngLayer


class TestnngLayer(unittest.TestCase):
    def test_init(self):
        """Testing the Layer init function

        """
        # Test of the nng_Layer
        test_layer = nngLayer(Dense(1))
        print(f"{test_layer.name = }")
        print(f"{test_layer.layer.name = }")
        test_layer(keras.Input((1, 28)))

    def test_add_neurons(self):
        """Testing the add_neurons function

        """
        test_layer = nngLayer(Dense(1))
        test_layer(keras.Input((1, 28)))
        print(test_layer.get_weights()[0])

        add_weights = np.ones((28, 3))
        add_bias = np.ones((3,))

        print(test_layer.name)

        test_layer.add_neurons(3, add_weights=add_weights,
                               add_bias=add_bias, in_lay=True)

        test_layer(keras.Input((1, 28)))

        print(test_layer.name)

        print(test_layer.get_weights()[0])


if __name__ == "__main__":
    unittest.main()
