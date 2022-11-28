import unittest
from unittest import TestCase

from nng.cfg import DatasetConfig, NNGConfig
from nng.dataloader import Dataloader
from nng.growingneuralnetwork import NNG


class Testnng(TestCase):
    def test_growing_neural_network(self):
        """Testing the growing_neural_network function

        """
        data_loader = Dataloader(DatasetConfig())
        training_ds, validation_ds, test_ds = data_loader.load_data()
        nng_config = NNGConfig.from_dict(
            {
                "seed": 60,
                "epochs": 2,
                "max_epochs": 2,
                "wandb": False,
                "plot": False
            }
        )
        nng = NNG(
            nng_cfg=nng_config,
            valid_ds=validation_ds,
            train_ds=training_ds,
            test_ds=test_ds,
            max_training=2
        )

        nng.neural_network_growth()

    def test_growing_neural_network_with_layergrowth(self):
        """Testing the neural_network_growth function

        """
        data_loader = Dataloader(DatasetConfig())
        training_ds, validation_ds, test_ds = data_loader.load_data()
        nng_config = NNGConfig.from_dict(
            {
                "where_type": "add_n_with_random_layer",
                "epochs": 2,
                "max_epochs": 2,
                "wandb": False,
                "plot": False
            }
        )
        nng = NNG(
            nng_cfg=nng_config,
            valid_ds=validation_ds,
            train_ds=training_ds,
            test_ds=test_ds,
            max_training=2
        )
        nng.neural_network_growth()


if __name__ == '__main__':
    unittest.main()
