import logging
import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

from nng.cfg import DatasetConfig
from nng.dataloader import Dataloader
from nng.grow import models
from nng.grow.models import create_nng
from nng.grow.modeltransformer import ModelTransformer
from nng.grow.modeltransformer import ModelTransformer as MTrafo
from nng.train import Trainer

logging.getLogger().setLevel(logging.INFO)


class TestModelTransformer(unittest.TestCase):

    def test_add_neurons(self):
        """Testing the add_neurons function

        """
        transformer = ModelTransformer('Adam',
                                       0.0,
                                       'SparseCategoricalCrossentropy',
                                       'SparseCategoricalAccuracy')

        model = create_nng(model_type='mlp', in_dim=784, out_dim=10)
        model(keras.Input((1, 28, 28)))
        print(model)

        transformer.grow_neuron(
            grow_neu_type='adding',
            model=model,
            n_neu=30,
            lay_idx=1,
            how_type='random_baseline',
            normalized=False
        )

        print(model)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=1)

    def test_grow_layer_baseline(self):
        """Testing the grow_layer_baseline function

        """
        transformer = ModelTransformer('Adam',
                                       0.0,
                                       'SparseCategoricalCrossentropy',
                                       'SparseCategoricalAccuracy')

        model = create_nng(model_type='mlp', in_dim=784, out_dim=10)
        model(keras.Input((1, 28, 28)))
        print(model)

        new_model = transformer.grow_layer(
            grow_lay_type='baseline',
            model=model,
            n_neu=1000,
            lay_idx=1,
            how_type='random_baseline',
            activation_function='relu',
            data_shape=(1, 28, 28),
            normalized=False
        )

        print(new_model)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        new_model.compile(optimizer='adam',
                          loss=loss_fn,
                          metrics=['accuracy'])
        new_model.fit(x_train, y_train, epochs=1)

    def test_adding_it_gradmax(self):
        """ adding_it_gradmax...
            - ...adds n_neu neurons to the layer specified by the `lay_idx`
            - For this the weights of the `lay_idx` and the `lay_idx +1` layer
                are changed.
            - The gradmax approach is to choose the init weights such that a
                certain constraint objective function is maximized. Here this
                maximization is realized via an iterative approach (projected
                gradient descent).
            - For this first the new weights in the layers are init with random
                entries which are then changed using backpropagation. 
            - In order to leave the output of the network unchanged after the
                growth the [new] weights in the in or out layer can be set to
                zero. Which layer is init with the gradmax strategy is set by
                the `gm_update_mode` parameter.

            - In this test we try the iterative gradmax update multiple times
                with the different `gm_update_mode`s. The the original and
                learned weight matrices are displayed for manual comparison
                (the old weight matrix needs to be present in the new one) and
                assessment.
        """
        data_loader = Dataloader(DatasetConfig())
        training_ds, validation_ds, test_ds = data_loader.load_data()
        print('\n\n\n\n\n\n gm_update_mode="in"==========\n\n\n\n\n\n')

        model = models.create_nng(model_type='mlp', out_dim=10, arch=[5, 5])
        model(keras.Input((1, 28, 28)))
        print(model)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

        n_neu = 50
        lay_idx = 2

        mtrafo = MTrafo()

        old_wghts_in = model.layers[lay_idx].get_weights()
        old_wghts_out = model.layers[lay_idx + 1].get_weights()

        kwargs = dict()

        mtrafo.grow_neuron(grow_neu_type='adding_it_gradmax',
                           lay_idx=lay_idx,
                           n_neu=n_neu,
                           model=model,
                           loss_function='SparseCategoricalCrossentropy',
                           gm_update_mode='in',
                           optimizer='Adam',
                           gm_lr=0.01,
                           gm_constraint=None,
                           gm_batch_size=1_000,
                           gm_max_it=75,
                           train_data=training_ds,
                           normalized=False,
                           **kwargs)

        model(keras.Input((1, 28, 28)))
        print(model)

        new_wghts_in = model.layers[lay_idx].get_weights()
        new_wghts_out = model.layers[lay_idx + 1].get_weights()

        print('RESULTS 1')

        print('\n\n\nold_wghts_in\n\n\n', old_wghts_in[0].T)
        print('\n\n\nold_wghts_in\n\n\n', old_wghts_in[0].T.shape)
        print('\n\n\nnew_wghts_in\n\n\n', new_wghts_in[0].T)
        print('\n\n\nnew_wghts_in\n\n\n', new_wghts_in[0].T.shape)
        print('\n\n\nold_wghts_out\n\n\n', old_wghts_out[0])
        print('\n\n\nold_wghts_out\n\n\n', old_wghts_out[0].shape)
        print('\n\n\nnew_wghts_out\n\n\n', new_wghts_out[0])
        print('\n\n\nnew_wghts_out\n\n\n', new_wghts_out[0].shape)

        print('\n\n\n\n\n\ngm_update_mode="out" ==========\n\n\n\n\n\n')

        model = models.create_nng(model_type='mlp', out_dim=10, arch=[5, 5])
        model(keras.Input((1, 28, 28)))
        print(model)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

        n_neu = 50
        lay_idx = 2

        mtrafo = MTrafo()

        old_wghts_in = model.layers[lay_idx].get_weights()
        old_wghts_out = model.layers[lay_idx + 1].get_weights()

        kwargs = dict()

        mtrafo.grow_neuron(grow_neu_type='adding_it_gradmax',
                           lay_idx=lay_idx,
                           n_neu=n_neu,
                           model=model,
                           loss_function='SparseCategoricalCrossentropy',
                           gm_update_mode='out',
                           optimizer='Adam',
                           gm_lr=0.01,
                           gm_constraint=None,
                           gm_batch_size=1_000,
                           gm_max_it=75,
                           train_data=training_ds,
                           normalized=False,
                           **kwargs)

        model(keras.Input((1, 28, 28)))
        print(model)

        new_wghts_in = model.layers[lay_idx].get_weights()
        new_wghts_out = model.layers[lay_idx + 1].get_weights()

        print('RESULTS 2')

        print('\n\n\nold_wghts_in\n\n\n', old_wghts_in[0].T)
        print('\n\n\nold_wghts_in\n\n\n', old_wghts_in[0].T.shape)
        print('\n\n\nnew_wghts_in\n\n\n', new_wghts_in[0].T)
        print('\n\n\nnew_wghts_in\n\n\n', new_wghts_in[0].T.shape)
        print('\n\n\nold_wghts_out\n\n\n', old_wghts_out[0])
        print('\n\n\nold_wghts_out\n\n\n', old_wghts_out[0].shape)
        print('\n\n\nnew_wghts_out\n\n\n', new_wghts_out[0])
        print('\n\n\nnew_wghts_out\n\n\n', new_wghts_out[0].shape)

        print('\n\n\n\n\n\gm_update_mode="both" ==========\n\n\n\n\n\n')

        model = models.create_nng(model_type='mlp', out_dim=10, arch=[5, 5])
        model(keras.Input((1, 28, 28)))
        print(model)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

        n_neu = 50
        lay_idx = 2

        mtrafo = MTrafo()

        old_wghts_in = model.layers[lay_idx].get_weights()
        old_wghts_out = model.layers[lay_idx + 1].get_weights()

        kwargs = dict()

        mtrafo.grow_neuron(grow_neu_type='adding_it_gradmax',
                           lay_idx=lay_idx,
                           n_neu=n_neu,
                           model=model,
                           loss_function='SparseCategoricalCrossentropy',
                           gm_update_mode='both',
                           optimizer='Adam',
                           gm_lr=0.01,
                           gm_constraint=None,
                           gm_batch_size=1_000,
                           gm_max_it=75,
                           train_data=training_ds,
                           normalized=False,
                           **kwargs)

        model(keras.Input((1, 28, 28)))
        print(model)

        new_wghts_in = model.layers[lay_idx].get_weights()
        new_wghts_out = model.layers[lay_idx + 1].get_weights()

        print('RESULTS 3')

        print('\n\n\nold_wghts_in\n\n\n', old_wghts_in[0].T)
        print('\n\n\nold_wghts_in\n\n\n', old_wghts_in[0].T.shape)
        print('\n\n\nnew_wghts_in\n\n\n', new_wghts_in[0].T)
        print('\n\n\nnew_wghts_in\n\n\n', new_wghts_in[0].T.shape)
        print('\n\n\nold_wghts_out\n\n\n', old_wghts_out[0])
        print('\n\n\nold_wghts_out\n\n\n', old_wghts_out[0].shape)
        print('\n\n\nnew_wghts_out\n\n\n', new_wghts_out[0])
        print('\n\n\nnew_wghts_out\n\n\n', new_wghts_out[0].shape)

    def test_adding_auto_lite(self):
        """ adding_auto_lite...
            - ...adds n_neu neurons to the layer specified by the `lay_idx`
            - For this the weights of the `lay_idx` and the `lay_idx +1` layer
                are changed.
            - The random_baseline is used to initialize the weights.
            - The AdamInit approach is training only the new weights which are
                added.
              The training is of the new weights is only done max 10 epochs,
              or untilthe training loss is <= as the training loss before
              adding the weights.
        """

        print('\n\n\n\n\n\nOUT ==========\n\n\n\n\n\n')

        model = models.create_nng(model_type='mlp', out_dim=10, arch=[5, 5])
        model(keras.Input((1, 28, 28)))
        print(model)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

        n_neu = 50
        lay_idx = 2

        data_loader = Dataloader(DatasetConfig())
        training_ds, validation_ds, test_ds = data_loader.load_data()
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=["accuracy"])
        train = Trainer(model, training_ds, validation_ds,
                        12, wandb=False)
        history, model = train.train(model)

        mtrafo = MTrafo()

        old_wghts_in = model.layers[lay_idx].get_weights()
        old_wghts_out = model.layers[lay_idx + 1].get_weights()

        kwargs = dict()

        mtrafo.grow_neuron(grow_neu_type='adding_auto_lite',
                           lay_idx=lay_idx,
                           n_neu=n_neu,
                           model=model,
                           loss_function='SparseCategoricalCrossentropy',
                           hw_update_mode='in',
                           optimizer='Adam',
                           hw_lr=0.01,
                           hw_constraint=None,
                           hw_batch_size=128,
                           hw_max_it=75,
                           history=history,
                           train_data=training_ds,
                           normalized=False,
                           **kwargs)

        model(keras.Input((1, 28, 28)))
        print(model)

        new_wghts_in = model.layers[lay_idx].get_weights()
        new_wghts_out = model.layers[lay_idx + 1].get_weights()

        print('stuff')

        print('\n\n\nold_wghts_in\n\n\n', old_wghts_in[0].T)
        print('\n\n\nold_wghts_in\n\n\n', old_wghts_in[0].T.shape)
        print('\n\n\nnew_wghts_in\n\n\n', new_wghts_in[0].T)
        print('\n\n\nnew_wghts_in\n\n\n', new_wghts_in[0].T.shape)
        print('\n\n\nold_wghts_out\n\n\n', old_wghts_out[0])
        print('\n\n\nold_wghts_out\n\n\n', old_wghts_out[0].shape)
        print('\n\n\nnew_wghts_out\n\n\n', new_wghts_out[0])
        print('\n\n\nnew_wghts_out\n\n\n', new_wghts_out[0].shape)

    def test_adding_firefly(self):
        """ This is a function for testing firefly How method.

        """
        model = models.create_nng(model_type='mlp', out_dim=10, arch=[5, 5])
        model(keras.Input((1, 28, 28)))
        print(model)
        data_loader = Dataloader(DatasetConfig())
        training_ds, validation_ds, test_ds = data_loader.load_data()

        n_neu = 50
        lay_idx = 2

        mtrafo = MTrafo()

        old_wghts_in = model.layers[lay_idx].get_weights()
        old_wghts_out = model.layers[lay_idx + 1].get_weights()

        kwargs = dict()

        mtrafo.grow_neuron(grow_neu_type='adding_firefly',
                           lay_idx=lay_idx,
                           n_neu=n_neu,
                           model=model,
                           loss_function='SparseCategoricalCrossentropy',
                           gm_update_mode='in',
                           optimizer='Adam',
                           gm_lr=0.01,
                           gm_constraint=None,
                           gm_batch_size=1_000,
                           gm_max_it=75,
                           train_data=training_ds,
                           normalized=False,
                           **kwargs)

        model(keras.Input((1, 28, 28)))
        print(model)

        new_wghts_in = model.layers[lay_idx].get_weights()
        new_wghts_out = model.layers[lay_idx + 1].get_weights()

        print('stuff')

        print('\n\n\nold_wghts_in\n\n\n', old_wghts_in[0])
        print('\n\n\nold_wghts_in\n\n\n', old_wghts_in[0].shape)
        print('\n\n\nnew_wghts_in\n\n\n', new_wghts_in[0])
        print('\n\n\nnew_wghts_in\n\n\n', new_wghts_in[0].shape)
        print('\n\n\nold_wghts_out\n\n\n', old_wghts_out[0])
        print('\n\n\nold_wghts_out\n\n\n', old_wghts_out[0].shape)
        print('\n\n\nnew_wghts_out\n\n\n', new_wghts_out[0])
        print('\n\n\nnew_wghts_out\n\n\n', new_wghts_out[0].shape)

        print('\n\n\n\n\n\nOUT ==========\n\n\n\n\n\n')

        model = models.create_nng(model_type='mlp', out_dim=10, arch=[5, 5])
        model(keras.Input((1, 28, 28)))
        print(model)

        n_neu = 50
        lay_idx = 2

        mtrafo = MTrafo()

        old_wghts_in = model.layers[lay_idx].get_weights()
        old_wghts_out = model.layers[lay_idx + 1].get_weights()

        kwargs = dict()

        mtrafo.grow_neuron(grow_neu_type='adding_firefly',
                           lay_idx=lay_idx,
                           n_neu=n_neu,
                           model=model,
                           loss_function='SparseCategoricalCrossentropy',
                           gm_update_mode='out',
                           optimizer='Adam',
                           gm_lr=0.01,
                           gm_constraint=None,
                           gm_batch_size=1_000,
                           gm_max_it=75,
                           train_data=training_ds,
                           normalized=False,
                           **kwargs)

        model(keras.Input((1, 28, 28)))
        print(model)

        new_wghts_in = model.layers[lay_idx].get_weights()
        new_wghts_out = model.layers[lay_idx + 1].get_weights()

        print('stuff')

        print('\n\n\nold_wghts_in\n\n\n', old_wghts_in[0].T)
        print('\n\n\nold_wghts_in\n\n\n', old_wghts_in[0].T.shape)
        print('\n\n\nnew_wghts_in\n\n\n', new_wghts_in[0].T)
        print('\n\n\nnew_wghts_in\n\n\n', new_wghts_in[0].T.shape)
        print('\n\n\nold_wghts_out\n\n\n', old_wghts_out[0])
        print('\n\n\nold_wghts_out\n\n\n', old_wghts_out[0].shape)
        print('\n\n\nnew_wghts_out\n\n\n', new_wghts_out[0])
        print('\n\n\nnew_wghts_out\n\n\n', new_wghts_out[0].shape)

    def test_grow_lay_idempotent(self):
        '''
            This test is validating that the output of a model is not chanching
            after adding a idempotent layer.
        '''
        model = models.create_nng(model_type='mlp', out_dim=10, arch=[5, 5])
        model(keras.Input((1, 28, 28)))
        print(model)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

        n_neu = 50
        lay_idx = 2

        data_loader = Dataloader(DatasetConfig())
        # 128, 42, [80, 19, 1], 'mnist')
        training_ds, validation_ds, test_ds = data_loader.load_data()
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=["accuracy"])
        train = Trainer(model, training_ds, validation_ds,
                        12, wandb=False)
        history, model = train.train(model)

        mtrafo = MTrafo()

        kwargs = dict()

        data = validation_ds.take(1)

        elem = training_ds.element_spec
        data_shape = elem[0].shape
        data_shape = [el for el in data_shape if el is not None]
        data_shape.reverse()

        prediction = model.predict(data)
        model = mtrafo.grow_layer(
            grow_lay_type='idempotent',
            model=model,
            n_neu=5,
            lay_idx=3,
            how_type='autogrow_zeroinit',
            activation_function='relu',
            data_shape=data_shape,
            **kwargs
        )

        prediction2 = model.predict(data)
        self.assertTrue(np.array_equal(prediction, prediction2))


if __name__ == "__main__":
    unittest.main()
