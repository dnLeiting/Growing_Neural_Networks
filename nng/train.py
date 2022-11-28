import logging

import tensorflow as tf
from wandb.keras import WandbCallback

logger = logging.getLogger(__name__)


class Trainer:
    """Implements model training methods.

    Attr:
        model (tf.keras.Model): Model to train.
        train_epochs (int): Number of epochs the model should be trained.
        ds_train (tf.data.Dataset): Training dataset.
        ds_test (tf.data.Dataset): Testing dataset.
        callbacks (tf.keras.callbacks): Callback for training monitoring with Wandb.
    """

    def __init__(self, model, train_ds, test_ds, epochs=10, wandb=True):
        self.model = model
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.train_epochs = epochs
        self.callbacks = [WandbCallback()] if wandb else []

    def train(self, model) -> tuple[tf.keras.callbacks.History, tf.keras.Model]:
        """Function to train a model

        Attr:
            - model (tf.keras.Model)  :  Model to train.

        Returns:
            - model (tf.keras.Model)    :   trained model
            - history (tf.keras.callbacks.History)  :   trained history
        """
        logger.info('train ' + str(self.train_epochs) + ' number of epochs')
        history = model.fit(
            self.train_ds,
            epochs=self.train_epochs,
            validation_data=self.test_ds,
            verbose=0,
            callbacks=self.callbacks,
        )
        return history, model
