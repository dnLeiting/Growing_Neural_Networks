import unittest

import tensorflow as tf

from nng.cfg import DatasetConfig
from nng.dataloader import Dataloader
from nng.grow.models import create_nng
from nng.grow.whentogrow import get_when_to_grow
from nng.train import Trainer


class TestPredefinedWhen(unittest.TestCase):
    def test_when_to_grow(self):
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
        when = get_when_to_grow(when_type='predefined',
                                model=model, max_training=5)
        train = Trainer(model, training_ds, validation_ds, 4, wandb=False)
        history, model = train.train(model)
        training_history = [history]
        total_training_epochs = 16
        for i in range(int(total_training_epochs / 4)):
            if when.when_to_grow(history=training_history, flops_history=None):
                print("grow")
                pass
            history_intermediate, model = train.train(model)
            training_history.append(history_intermediate)
        self.assertEqual(len(training_history), 5)


if __name__ == "__main__":
    unittest.main()
