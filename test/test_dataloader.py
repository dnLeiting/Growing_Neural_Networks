import unittest
from unittest import TestCase

from nng.cfg import DatasetConfig
from nng.dataloader import Dataloader


class TestDataloader(TestCase):

    def test_load_data(self):
        """Function that tests the method "load_data" in the Dataloader class

        """
        data_loader = Dataloader(DatasetConfig())
        training_ds, validation_ds, test_ds = data_loader.load_data()
        self.assertTrue(len(training_ds) > 0)
        self.assertTrue(len(validation_ds) > 0)

    def test_split_data(self):
        """Function that tests the method "split_data" in the Dataloader class

        """
        data_loader = Dataloader(DatasetConfig.from_dict({"seed": 90}))
        training_ds, validation_ds, test_ds = data_loader.load_data()
        print(len(list(training_ds)))
        print(len(list(validation_ds)))
        print(len(list(test_ds)))
        self.assertTrue(len(list(training_ds)) > len(
            list(validation_ds)))


if __name__ == "__main__":
    unittest.main()
