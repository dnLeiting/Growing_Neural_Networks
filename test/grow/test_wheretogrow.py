import logging
import unittest
from dataclasses import dataclass, field
from unittest import TestCase

import keras
import tensorflow as tf

from nng.cfg import DatasetConfig
from nng.dataloader import Dataloader
from nng.grow.models import create_nng
from nng.grow.modeltransformer import ModelTransformer
from nng.grow.wheretogrow import get_where_to_grow
from nng.train import Trainer

logging.getLogger().setLevel(logging.INFO)


@dataclass
class DatasetConfig():
    dataset: str = 'mnist'
    num_classes: int = 10
    batch_size: int = 128
    split_rate: list[int] = field(default_factory=lambda: [80, 19, 1])
    seed: int = 42


class TestPredefinedWhere(TestCase):
    def test_predefined(self):
        """Function that tests the method "when_to_grow" of the PredefinedWhen
            class

        """
        data_loader = Dataloader(DatasetConfig())
        training_ds, validation_ds, test_ds = data_loader.load_data()

        model = create_nng(model_type='mlp', out_dim=10)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        where = get_where_to_grow(where_type="add_predefined",
                                  model=model, )
        train = Trainer(model, training_ds, validation_ds, 4, wandb=False)
        history, model = train.train(model)
        print(where.where_to_grow(model=model, neuron_growth=5))

    def test_predefined_with_layer(self):
        data_loader = Dataloader(DatasetConfig())
        training_ds, validation_ds, test_ds = data_loader.load_data()

        model = create_nng(model_type='mlp', out_dim=10)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        where = get_where_to_grow(where_type="add_n_with_random_layer",
                                  model=model, )
        train = Trainer(model, training_ds, validation_ds, 4, wandb=False)
        history, model = train.train(model)
        history_all = []
        history_all.append(history)
        print(where.where_to_grow(model=model,
                                  neuron_growth=5, history=history_all))


class TestTriggerWhere(TestCase):
    def test_where_to_grow(self):
        """Testing the where_to_grow function

        """
        tf.random.set_seed(1)

        # load dataset
        data_loader = Dataloader(DatasetConfig())
        training_ds, validation_ds, test_ds = data_loader.load_data()

        for mode in ['orthog_w', 'efctdim_w', 'orthog_h', 'efctdim_h']:
            # create model
            model = create_nng(model_type='mlp', in_dim=784, out_dim=10)

            model.data_shape = (1, 28, 28)

            # NOTE needed for the init of the lay wghts
            model(keras.Input(model.data_shape))

            # compile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], )

            where = get_where_to_grow(where_type='Trigger',
                                      trigger_type=mode,
                                      model=model,
                                      with_lays=False,
                                      training_ds=training_ds,
                                      data_shape=model.data_shape)

            print('\n', where.where_to_grow(
                model=model,
                neuron_growth=None,
                history=None, ), '\n')

    def test_where_to_grow_w_lay_growth(self):
        """ Test Layer-Growth """
        tf.random.set_seed(1)

        # load dataset
        data_loader = Dataloader(DatasetConfig())
        training_ds, validation_ds, test_ds = data_loader.load_data()

        # create model
        model = create_nng(model_type='mlp', in_dim=784, out_dim=10)

        data_shape = (1, 28, 28)

        # NOTE needed for the init of the lay wghts
        model(keras.Input(data_shape))

        # compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], )

        trigger_type = 'orthog_h'

        where = get_where_to_grow(where_type='Trigger',
                                  trigger_type=trigger_type,
                                  model=model,
                                  thr_rep=0.1,
                                  with_lays=True,
                                  exp_thr=1,
                                  data_shape=data_shape,
                                  training_ds=training_ds)

        transformer = ModelTransformer('Adam',
                                       0.8,
                                       'SparseCategoricalCrossentropy',
                                       'SparseCategoricalAccuracy')

        assert hasattr(model.layers[1], "seed_size") \
               and model.layers[1].seed_size == model.layers[1].layer.units, \
            "Not correctly init with trigger_thr method!"

        for lay in model.layers[1:-1]:
            print(lay.seed_size)
            print(lay.init_thr)

        n = 3

        for _ in range(1, n + 1):
            print("-" * 10)
            print(f"\n{_}/{n}\n")
            print(("-" * 10) + "\n\n")

            new_neus, new_lays = where.where_to_grow(
                model=model,
                neuron_growth=None,
                history=None, )

            print("\n{}\n{}\n".format(new_neus, new_lays))

            for lay_idx, n_neu in enumerate(new_neus):
                if n_neu == 0:
                    continue  # nothing to grow

                # Growing

                if lay_idx in new_lays:
                    # Grow Layer
                    model = transformer.grow_layer(
                        grow_lay_type="baseline",
                        model=model,
                        n_neu=n_neu,
                        lay_idx=lay_idx,
                        how_type="random_baseline",
                        activation_function="relu",
                        data_shape=(1, 28, 28),
                    )

                else:

                    # Grow Neurons
                    transformer.grow_neuron(
                        grow_neu_type="adding",
                        model=model,
                        n_neu=n_neu,
                        lay_idx=lay_idx,
                        how_type='random_baseline',
                        train_data=None,
                        loss_function=None,
                    )

                # needed so every layer and in particular its weights
                # are  initialised
                model(keras.Input(data_shape))
                print(model)

            # The trigger thrs. needs to be set to newly grown layers
            if True and new_lays != []:
                where.thr_init(model=model,
                               trigger_type=trigger_type,
                               training_ds=training_ds)


if __name__ == "__main__":
    unittest.main()
