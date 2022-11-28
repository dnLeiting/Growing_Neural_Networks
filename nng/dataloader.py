import logging

import tensorflow as tf
import tensorflow_datasets as tfds

from .cfg import DatasetConfig

logger = logging.getLogger(__name__)


class Dataloader:
    """Implements dataset loading methods.

    Description:

    Attr:
        - cfg (DatasetConfig)  :  Config containing all the information for training. This includes dataset, seed, and splitting-rate
    """

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg

    def load_data(self):
        """Function for loading the a dataset

        Returns:
            (tf.data.Dataset)
        """

        supported_datasets = [
            'mnist',
            'fashion_mnist',
            'oxford_flowers102'
        ]

        if self.cfg.dataset not in supported_datasets:
            raise ValueError('Invalid dataset.')

        assert self.cfg.split_rate[0] + \
               self.cfg.split_rate[1] + self.cfg.split_rate[2] == 100

        (training_ds, validation_ds, test_ds), ds_info = tfds.load(
            'mnist',
            split=[f'train[:{self.cfg.split_rate[0]}%]',
                   f'train[{self.cfg.split_rate[0]}%:{self.cfg.split_rate[0] + self.cfg.split_rate[1]}%]',
                   f'train[{self.cfg.split_rate[0] + self.cfg.split_rate[1]}%:]'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            read_config=tfds.ReadConfig(
                shuffle_seed=self.cfg.seed))

        def normalize_img(image, label):
            """Normalizes images: `uint8` -> `float32`."""
            return tf.cast(image, tf.float32) / 255., label

        training_ds = training_ds.map(
            normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        training_ds = training_ds.batch(self.cfg.batch_size)
        training_ds = training_ds.prefetch(tf.data.AUTOTUNE)

        validation_ds = validation_ds.map(
            normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        validation_ds = validation_ds.batch(self.cfg.batch_size)
        validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)

        test_ds = test_ds.map(
            normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.cfg.batch_size)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

        logger.info('Training dataset batches ' + str(len(training_ds)))
        logger.info('Validation dataset batches ' + str(len(validation_ds)))
        logger.info('Test dataset batches ' + str(len(test_ds)))

        return training_ds, validation_ds, test_ds
