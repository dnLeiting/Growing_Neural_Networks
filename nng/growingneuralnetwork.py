import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
import wandb
from tensorflow import keras

from nng.analysis_utils import log_analysis, get_flops
from nng.grow.models import create_nng
from nng.grow.modeltransformer import ModelTransformer
from nng.grow.whentogrow import get_when_to_grow
from nng.grow.wheretogrow import get_where_to_grow
from nng.train import Trainer
from .cfg import NNGConfig

logger = logging.getLogger(__name__)


class NNG:
    """Implements growing neural network pipeline.

    Attr:
      - dataset (str)   :  Name of the dataset to use.
      - epochs (int)    :   Epochs to train before calling the when function.
      - max_epochs (int) :  Max number of epochs a neural network will train.
                            If set to 0 the training process will stop without
                            growing the nn.
      - metric (str)    :   Defines the metrics to be used for training the
                            model.
      - optimiser (str) :   Defines the optimiser to be used for training the
                            model.
      - batch_size (int)    :   Batch size to be used.
      - split_rate (list)   :   Split rate to be used for training, validation,
                                and test data.
      - model_type (str)    :   Type of neural network to be initialised.
      - loss_function (str) :   Defines the loss function to be used for
                                training the model.
      - init_architecture (list)    :   Initial architecture layer structure.
      - activation_function (list)  :   Activation function to be used for the
                                        network.
      - seed (int)  :   Seed to be used for the generation of batches.
      - when (str)  :   Methode to decide on when to grow.
      - where (str) :   Methode to decide on where to grow.
      - how (str)   :   Methode to decide on how to grow.
      - learning_step (float)   :   Defines the training steps to be used for
                                    training the model.
    """

    def __init__(
            self,
            nng_cfg: NNGConfig,
            train_ds,
            valid_ds,
            test_ds=None,
            **kwargs
    ):
        self.nng_cfg = nng_cfg
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.kwargs = kwargs

        labels = list(map(lambda x: x[1], train_ds))
        self.num_classes = len(set(np.array(tf.concat(labels, axis=0))))

        elem = self.train_ds.element_spec
        data_shape = elem[0].shape
        self.data_shape = [el for el in data_shape if el is not None]
        self.data_shape.reverse()

    def neural_network_growth(self):
        """Function for running the growing neural network pipeline.

        Description: This function is the core of the growing neural network pipeline.
        Config files are read and the growing/training loop with When, Where and How is executed.
        """

        # Init the model

        optimizer = getattr(keras.optimizers, self.nng_cfg.optimiser)
        loss_fn = getattr(keras.losses, self.nng_cfg.loss_function)
        metric = getattr(keras.metrics, self.nng_cfg.metric)

        def compile_func(model, optimizer):
            model.compile(optimizer=optimizer,
                          loss=loss_fn(from_logits=True),
                          metrics=[metric()], )

        def compile_func_initial(model):
            model.compile(optimizer=optimizer(self.nng_cfg.learning_step),
                          loss=loss_fn(from_logits=True),
                          metrics=[metric()], )

        if self.nng_cfg.model_type == 'MLP':
            model = create_nng(model_type=self.nng_cfg.model_type,
                               out_dim=self.num_classes,
                               arch=self.nng_cfg.init_architecture)

            # the optimizer (as it is done in the gradmax code!) If this is
            # the case we might need to create a custom training loop in order
            # to access the actual optimizer object with its current internal
            # parameters
            compile_func_initial(model)

        else:
            raise NotImplemented("CNN needs to be figured out.")

        # Data structure setup
        train = Trainer(model, self.train_ds, self.valid_ds,
                        self.nng_cfg.epochs, self.nng_cfg.wandb)

        when = get_when_to_grow(when_type=self.nng_cfg.when_type,
                                model=model, **self.kwargs)

        where = get_where_to_grow(where_type=self.nng_cfg.where_type,
                                  model=model,
                                  training_ds=self.train_ds,
                                  data_shape=self.data_shape,
                                  **self.kwargs)

        transformer = ModelTransformer(self.nng_cfg.optimiser,
                                       self.nng_cfg.learning_step,
                                       self.nng_cfg.loss_function,
                                       self.nng_cfg.metric)
        training_history = []
        flops_history = []
        flops_per_batch = 0

        if self.nng_cfg.wandb and self.nng_cfg.plot:
            if self.test_ds is not None:
                log_analysis(self.test_ds, model, commit=False)
            else:
                log_analysis(self.valid_ds, model, commit=False)

        # Training and Growing Loop
        for current_epoch in range(int(self.nng_cfg.max_epochs)):
            start = datetime.now()
            flops_history.append(get_flops(model, self.data_shape))
            end = datetime.now()
            logger.info("FLOPS in current model: {0}".format(
                flops_history[-1]))

            if self.nng_cfg.wandb:
                wandb.run.log({"flops": flops_history[-1]},
                              step=wandb.run.step,
                              commit=False)
                wandb.run.log(
                    {"time2calculate_flops": (end - start).microseconds},
                    step=wandb.run.step,
                    commit=False)

            # Training Step
            start = datetime.now()
            history, model = train.train(model)
            optimizer = model.optimizer
            end = datetime.now()
            training_history.append(history)
            flops_per_batch += flops_history[-1] * self.nng_cfg.epochs

            if self.nng_cfg.wandb:
                wandb.run.log({"time2train": (end - start).microseconds},
                              step=wandb.run.step, commit=False)

            logger.info(
                f"Epoch: ({current_epoch + 1})/{int(self.nng_cfg.max_epochs)}")
            logger.info(model)

            if self.nng_cfg.wandb and self.nng_cfg.plot:
                if self.test_ds is not None:
                    log_analysis(self.test_ds, model, commit=False)
                else:
                    log_analysis(self.valid_ds, model, commit=False)

            # Growing step
            # when to grow?
            start = datetime.now()
            when_bool = when.when_to_grow(
                history=training_history, flops_history=flops_history)
            end = datetime.now()

            if self.nng_cfg.wandb:
                wandb.run.log({"time2when": (end - start).microseconds},
                              step=wandb.run.step, commit=False)

            if when_bool and self.nng_cfg.growing:
                # where to grow?
                start = datetime.now()
                new_neus, new_lays = where.where_to_grow(
                    model=model,
                    neuron_growth=self.nng_cfg.neuron_growth,
                    history=training_history, )
                end = datetime.now()
                logger.info('new_neus' + str(new_neus))

                if self.nng_cfg.wandb:
                    wandb.run.log({"time2where": (end - start).microseconds},
                                  step=wandb.run.step,
                                  commit=False)

                for lay_idx, n_neu in enumerate(new_neus):

                    if n_neu == 0:
                        continue  # nothing to grow

                    # Growing

                    start = datetime.now()
                    if lay_idx in new_lays:
                        # Grow Layer
                        model = transformer.grow_layer(
                            grow_lay_type=self.nng_cfg.grow_lay_type,
                            model=model,
                            n_neu=n_neu,
                            lay_idx=lay_idx,
                            how_type=self.nng_cfg.how_type,
                            activation_function=self.nng_cfg.activation_function,
                            data_shape=self.data_shape,
                            normalized=self.nng_cfg.normalized,
                            **self.kwargs
                        )

                    else:

                        # Grow Neurons
                        transformer.grow_neuron(
                            grow_neu_type=self.nng_cfg.grow_neu_type,
                            model=model,
                            n_neu=n_neu,
                            lay_idx=lay_idx,
                            how_type=self.nng_cfg.how_type,
                            train_data=self.train_ds,
                            loss_function=self.nng_cfg.loss_function,
                            history=history,
                            normalized=self.nng_cfg.normalized,
                            **self.kwargs
                        )

                    end = datetime.now()

                    if self.nng_cfg.wandb:
                        wandb.run.log({"time2grow": (end - start).microseconds},
                                      step=wandb.run.step,
                                      commit=False)

                    # needed so every layer and in particular its weights
                    # are  initialised
                    model(keras.Input(self.data_shape))

                # The trigger thrs. needs to be set to newly grown layers
                if self.nng_cfg.where_type.lower() == 'trigger' \
                        and new_lays != []:
                    where.thr_init(
                        model=model,
                        training_ds=self.train_ds,
                        **self.kwargs)

                # new weights need to be init and recognized by the
                compile_func(model, optimizer)

                logging.info(
                    f"{current_epoch}/{self.nng_cfg.max_epochs} <-------")

                print("\n\n" + ("-" * 15) + "\n\n")
                print(f"\n EPOCH{current_epoch}/{self.nng_cfg.max_epochs}\n")
                print(f"\n{new_neus = }\n{new_lays = }\n")
                print(model)
                print("\n\n" + ("-" * 25) + "\n\n")

        history, model = train.train(model)
        training_history.append(history)
        flops_per_batch += flops_history[-1] * self.nng_cfg.epochs
        if self.nng_cfg.wandb:
            wandb.log({"Total Flops per batch": flops_per_batch}, commit=False)

        # Print plots
        if self.nng_cfg.wandb and self.nng_cfg.plot:
            if self.test_ds is not None:
                log_analysis(self.test_ds, model)
            else:
                log_analysis(self.valid_ds, model)
