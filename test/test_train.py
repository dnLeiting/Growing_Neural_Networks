import unittest
from unittest import TestCase

from tensorflow import keras

from nng.cfg import DatasetConfig
from nng.dataloader import Dataloader
from nng.grow.models import create_nng_mlp
from nng.train import Trainer


class TestTrain(TestCase):
    def test_train(self):
        """Function that tests the method "train" in the Train class

        """
        data_loader = Dataloader(DatasetConfig())
        training_ds, validation_ds, test_ds = data_loader.load_data()

        optimizer = getattr(keras.optimizers, 'Adam')
        loss_fn = getattr(keras.losses, 'SparseCategoricalCrossentropy')
        metric = getattr(keras.metrics, 'SparseCategoricalAccuracy')

        model = create_nng_mlp(in_dim=784, out_dim=10)
        model.compile(optimizer=optimizer(0.001),
                      loss=loss_fn(from_logits=True),
                      metrics=[metric()], )
        print(model)

        train = Trainer(model, training_ds, validation_ds, 5, wandb=False)
        history, model = train.train(model)
        self.assertEqual(len(history.history['loss']), 5)
        self.assertIsNotNone(model)

        print(model)


if __name__ == "__main__":
    unittest.main()
