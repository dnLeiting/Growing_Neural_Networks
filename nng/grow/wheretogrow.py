import logging
from collections import deque

import keras
import numpy as np
import numpy.linalg as la
import tensorflow as tf

from nng.grow.layers import nngLayer

logger = logging.getLogger(__name__)


class WhereToGrow:
    """Implements the where to grow methods.

    Attr:
      model (tf.keras.Model): Machine Learning Model.
      args (tf.keras.callbacks.History): Training History.
    """

    def __init__(self, **kwargs):
        pass

    def where_to_grow(self, **kwargs):
        raise NotImplementedError()


class GradMaxWhere(WhereToGrow):
    def where_to_grow(self) -> tuple[list, list]:
        raise NotImplementedError


class AutoGrowWhere(WhereToGrow):
    def __init__(self, model, **args):
        self.linked_list = deque()
        for id_, _ in enumerate(model.layers):
            self.linked_list.append(id_)
        self.linked_list.popleft()
        self.linked_list.pop()
        self.K = None
        self.tau = args.get("tau")
        self.J = args.get("J")
        self.where_autogrow_policy = args.get("where_autogrow_policy")
        self.neurons_to_grow = args.get("neurons_to_grow")
        logger.info("Set Where with methods: {0}, tau: {1}, J: {2}".format(
            self.where_autogrow_policy,
            self.tau,
            self.J
        ))

    def where_to_grow(self, model,
                      history: list, **kwargs) -> tuple[list, list]:
        """
        We calculate K as number of already trained epochs in the first time
        we run this where methods.

        Args:
            - history: (list) - list of losses in previous training steps.
            - kwargs (dict) :   Used to pipe the arguments into the selected
                method. This allows methods with different
                arguments to be called and allows also future
                implementations/changes.

        Returns:
            - new_neurons: (list) List of new neurons, which should be added to each layer
            - new_layer: (list) List of numbers. Each number represents a layer which should added new
        """
        if self.K is None:
            self.K = len(history)
            if not self.K:
                raise ValueError("Pass loss history with K epochs.")
        new_neurons = [0] * len(model.layers)
        # If stopping policy met than remove last layer grown
        if len(self.linked_list) > 0 and self.stopping_policy(history):
            popped_lay = self.linked_list.pop()
            logger.info("Remove layer: {0}".format(popped_lay))
        # If growing policy met then grow layer and put it at the end
        if len(self.linked_list) > 0 and self.growing_policy(history):
            lay_id = self.linked_list.popleft()
            new_neurons[lay_id] = self.neurons_to_grow
            logger.info("Grow layer {0} by {1}".format(
                lay_id,
                self.neurons_to_grow
            ))
            self.linked_list.append(lay_id)
        return new_neurons, []

    def growing_policy(self, loss_history: list) -> bool:
        """
            Should we grow the layer?
        """
        if self.where_autogrow_policy == 'ConvergentGrowth':
            if (
                    loss_history[-1 * self.K].history['val_loss'][0]
                    - loss_history[-1].history['val_loss'][0]
            ) < self.tau:
                return True
            else:
                return False
        elif self.where_autogrow_policy == 'PeriodicGrowth':
            return True
        else:
            raise ValueError("Wrong value for the where_autogrowth_parameter \
            choose between 'PeriodicGrowth' and 'ConvergentGrowth'")

    def stopping_policy(self, loss_history: list) -> bool:
        """
        Should we stop growing last layer?
        Did new neurons added to the previous layer added any new information?
        If not than we return True.
        """
        if self.where_autogrow_policy == 'ConvergentGrowth':
            loss_diff = loss_history[
                            -1 * self.K
                            ].history['val_loss'][0] - \
                        loss_history[-1].history['val_loss'][0]
            logger.info("Recent growth loss {0}".format(loss_diff))
            if loss_diff < self.tau:
                return True
            else:
                return False
        elif self.where_autogrow_policy == 'PeriodicGrowth':
            if self.J <= self.K:
                raise ValueError("J should be bigger than K.")
            if len(loss_history) < self.J:
                return True
            elif (
                    loss_history[-1 * self.J].history['val_loss'][0]
                    - loss_history[-1].history['val_loss'][0]
            ) < self.tau:
                return True
            else:
                return False
        else:
            raise ValueError("Wrong value for the where_autogrowth_parameter \
            choose between 'PeriodicGrowth' and 'ConvergentGrowth'")


class PredefinedWhere(WhereToGrow):
    def where_to_grow(self, model,
                      neuron_growth, **kwargs) -> tuple[list, list]:
        """Default function for deciding where to grow. Following a predefined stucture. New layers are not added.

        Args:
            - model (tf.keras.Model):   Model to grow
            - neuron_growth (int) : number of layers to add
            - history (list of tf.keras.callbacks.History)  : List of previous training history.
            - kwargs (dict) :   Used to pipe the arguments into the selected
                method. This allows methods with different
                arguments to be called and allows also future
                implementations/changes.

        Returns:
            - new_layers (list):  Returns the structure to be added to the model.
            - (list):  The index of the new layers.
        """
        del kwargs
        new_neurons = [0]
        for layer in model.layers[1:-1]:
            if isinstance(layer.layer, tf.keras.layers.Dense):
                new_neurons.append(neuron_growth)
            else:
                new_neurons.append(0)
        new_neurons.append(0)
        return new_neurons, []


class PredefinedWhereWithLayerGrowth(WhereToGrow):

    def where_to_grow(self, model,
                      neuron_growth, history, **kwargs) -> tuple[list, list]:
        """Default function for deciding where to grow with adding layers.

        Args:
            - model (tf.keras.Model):   Model to grow
            - neuron_growth (int) : number of layers to add
            - history (list of tf.keras.callbacks.History)  : List of previous training history.
            - kwargs (dict) :   Used to pipe the arguments into the selected
                method. This allows methods with different
                arguments to be called and allows also future
                implementations/changes.

        Returns:
            - new_layers (list):  Returns the structure to be added to the model.
            - (list):  The index of the new layers.
        """
        new_neurons = []
        new_layer = []
        if (len(model.layers) == 3):
            n_lay_idx = 1
            form_old_layer = 2
        else:
            n_lay_idx = np.random.choice(range(1, len(model.layers) - 2))
            form_old_layer = np.random.choice(range(2, len(model.layers) - 1))

        n_lay_neu = model.layers[form_old_layer].get_weights()[0].shape[0]
        logger.info('n_lay_idx ' + str(n_lay_idx))
        for lay_idx, layer in enumerate(model.layers[0:-1]):

            if isinstance(layer, nngLayer):
                # or isinstance(layer, tf.keras.layers.Conv2D):
                new_neurons.append(neuron_growth)
                if lay_idx == n_lay_idx:
                    # add the neurons for the new layer
                    new_neurons.append(n_lay_neu + neuron_growth)
                # add neurons to existing layers

            else:
                new_neurons.append(0)

        new_layer.append(n_lay_idx + 1)
        new_neurons.append(0)
        return new_neurons, new_layer


class TriggerWhere(WhereToGrow):
    def __init__(self, *, model: keras.Model,
                 trigger_type: str,
                 data_shape: tuple,
                 training_ds,
                 with_lays: bool = True,
                 exp_thr: int = 10,
                 thr_reg: float = 1.0,
                 **kwargs):
        """Implementatino based on 'When, where and how to add Neurons to ANNs"
            Maile et al.

        Args:
            model (keras.Model): Model to be grown
            trigger_type (str): ['orthog_w', 'orthog_h', 'efctdim_w', 
                                'efctdim_h'] set which metric is used 
                                1. effective dim or
                                2. orthogonality gab
                                with ether
                                * h: Post-activsion matrix
                                * w: Weight matrix
            data_shape (tuple): Dimensions of the data the model is trained on.
                                (Needed for layer init.)
            training_ds (_type_):   Dataset needed for computation of the post
                                    -activisions for the '_h' triggers.
            with_lays (bool, optional):     Whether LayerGenesis is also 
                                            performed or only NeuroGenesis.
                                            Defaults to True.
            exp_thr (int, optional):    'expansion threshold' percentage 
                                        determining how many neurons (relative
                                        to the seed size) can be added to a 
                                        layer before a new layer is grown
                                        instead. Defaults to 10.
            thr_rep (float, optional):  regularisation factor multiplied to the
                                        initial metric when setting the 
                                        init_thr attribute to the layers. 
                                        Defaults to 1.0.
        """

        assert 1 <= exp_thr, "Expansion threshold (exp_thrs) is a " \
                             "percentage and should be an integer in the range of 1 " \
                             "provided was: {}.".format(exp_thr)

        self.trigger_type = trigger_type
        self.training_ds = training_ds
        self.exp_thr = exp_thr
        self.with_lays = with_lays
        self.thr_reg = thr_reg
        self.data_shape = data_shape

        # Set the initail treshold
        model(keras.Input(self.data_shape))
        self.thr_init(model=model,
                      training_ds=training_ds,
                      trigger_type=trigger_type,
                      thr_reg=self.thr_reg,
                      with_lays=with_lays,
                      **kwargs)

    def where_to_grow(self, model, neuron_growth, history, **kwargs):
        """ Function used for initializing the thresholds needed for the
            triggers based on the paper 'When, where and how to add neurons to
            ANNs' by Maile et al.

            Args:
            - model (keras.Model): Network to be grown
            - trigger_type (str): describes the trigger type
            - thr_reg (float)
            - with_lays (bool)
            - kwargs (dict) :   Used to pipe the arguments into the selected
                            method. This allows methods with different
                            arguments to be called and allows also future
                            implementations/changes.

        """

        if "_w" in self.trigger_type.lower():
            extract_fn = self.get_wghts
        else:
            extract_fn = self.get_post_activitions
        if "orthog_" in self.trigger_type.lower():
            score_fn = self.orthog_gap
        else:
            score_fn = self.efct_dim

        mxt_list = extract_fn(model=model, training_ds=self.training_ds)

        assert len(mxt_list) == len(model.layers), "problem in extract_fn. {}"
        new_neus_num, new_lay_idx = [], []

        for i, w in enumerate(mxt_list):
            if w is None:
                # if lay is e.g. keras.layers.Flatten
                new_neus_num.append(0)
                continue

            # extract the threshold value
            try:
                thr = model.layers[i].init_thr
                if not 0 <= thr <= 1:
                    raise ValueError(
                        "The threshold needs to be contained in [0,1]."
                        "Found threshold {}.".format(thr))
            except Exception as e:
                raise AttributeError(
                    "Threshold not set? Got error: {}.".format(e))

            score = score_fn(w)
            logging.info(
                f"Layer: {i}, Score: {score}, Init/Thresh.: {thr}.")

            diff = score - thr

            logging.info(f"{diff=}, {diff*w.shape[1]=}")

            if diff <= 0:
                new_neus_num.append(0)
                continue

            # diff times num units in lay = num new neurons
            num = int(np.floor(diff * w.shape[1]))
            new_neus_num.append(num)

        assert len(new_neus_num) == len(model.layers), \
            "1. Num for new_nums not equal the number of layers in the net!"

        if self.with_lays:
            tmp_nn, tmp_nl = [], []
            for i, new_neus in enumerate(new_neus_num):
                if new_neus == 0:
                    tmp_nn.append(new_neus)
                    continue
                lay = model.layers[i]

                units = lay.layer.units
                if units >= np.ceil(lay.seed_size
                                    * (1 + self.exp_thr / 100)):
                    # expansion threshold has been hit dont grow new neuron but
                    # new layer instead
                    tmp_nn.append(0)  # no growth on this layer
                    lay.seed_size = units  # update so it can be grown again

                    # new layer at the idx after the current one
                    tmp_nl.append(len(tmp_nn))
                    # the new layer needs as many neurons as its predecessor
                    # else we would prune the network
                    tmp_nn.append(units)
                else:
                    tmp_nn.append(new_neus)

            new_neus_num, new_lay_idx = tmp_nn, tmp_nl

        assert len(new_neus_num) >= len(model.layers), \
            "2. Num for new_nums smaller than the number of layers in the net!"

        return new_neus_num, new_lay_idx

    # Metrics

    def orthog_gap(self, W: np.ndarray, **kwargs) -> np.float64:
        """Function that computes the orthogonality gap. See 
            - 'When, where, and how to add new neurons to ANNs' - Maile et al.
            - 'BATCH NORMALIZATION ORTHOGONALIZES REPRESENTATIONS IN DEEP
                RANDOM NETWORKS' - DANESHMAND et al.
        """
        tmp = la.norm(W.T @ W / la.norm(W, 'fro') ** 2
                      - np.eye(W.shape[1]) / W.shape[1], 'fro')
        return 1 - tmp

    def efct_dim(self, W: np.ndarray, efct_dim_thr: float = 0.01,
                 partial: bool = False, **kwargs) -> np.float64:
        """Function that computes the effective dimension. See
            - 'When, where, and how to add new neurons to ANNs' - Maile et al.
            - 'Understanding and preventing capacity loss in reinforcement
                learning' - Lyle et al.
        """
        _, s, _ = la.svd(W / np.sqrt(W.shape[1]))

        return (s > efct_dim_thr).sum() / W.shape[1]

    # Threshold init

    def thr_init(self, *, model: keras.Model,
                 trigger_type: str,
                 training_ds,
                 thr_reg: float,
                 with_lays: bool = True,
                 **kwargs):
        """ Function used for initializing the thresholds needed for the
            triggers based on the paper 'When, where and how to add neurons to
            ANNs' by Maile et al.

            Args:
            - model (keras.Model): Network to be grown
            - trigger_type (str): describes the trigger type
            - thr_reg (float)
            - with_lays (bool)
            - kwargs (dict) :   Used to pipe the arguments into the selected
                            method. This allows methods with different
                            arguments to be called and allows also future
                            implementations/changes.

        """
        supported_triggers = ['orthog_w', 'efctdim_w',
                              'orthog_h', 'efctdim_h']
        assert trigger_type in supported_triggers, \
            "{} not supported, supported are {}".format(trigger_type,
                                                        supported_triggers)

        if "_w" in trigger_type.lower():
            extract_fn = self.get_wghts
        else:
            extract_fn = self.get_post_activitions
        if "orthog_" in trigger_type.lower():
            score_fn = self.orthog_gap
        else:
            score_fn = self.efct_dim

        mxt_list = extract_fn(model=model,
                              training_ds=training_ds,
                              **kwargs)

        for i, m in enumerate(mxt_list):
            lay = model.layers[i]
            if m is None or hasattr(lay, 'thr_reg'):
                continue
            lay.init_thr = thr_reg * score_fn(m, **kwargs)
            if with_lays:
                lay.seed_size = lay.layer.units

                logging.info("trigger_init > {}: \n"
                             "      {} & {}".format(lay,
                                                    lay.seed_size,
                                                    lay.init_thr))

    # Extraction methods

    def get_wghts(self, *, model: keras.Model, **kwargs) -> list[np.ndarray]:
        del kwargs
        wght_list = []
        for i, lay in enumerate(model.layers):
            if not isinstance(lay, nngLayer) or i == len(model.layers) - 1:
                wght_list.append(None)
                continue
            w, _ = lay.get_weights()
            wght_list.append(w)
        return wght_list

    def get_post_activitions(self, *,
                             model: keras.Model,
                             training_ds: tf.data.Dataset,
                             **kwargs) -> list[np.ndarray]:
        """ Computes the postactivition matricies as described in Maile et al.
            'When, where and how to add Neurons to ANNs'

        Description:
            - For n data points we compute the activation at each layer and
                combine them into a (n x m) dimensional matrix, where m is the
                number of neurons in that layer.
            - In order to be able to compute the Effective Dimension we need
                n >= m thus we select n to be greater or equal than the number
                of neurons in the widest layer in the network.
            - To lighten the computation of e.g. the svd we shrink the matrices
                of much shallower layers.

        Args:
            - model (keras.Model): Network to be grown
            - training_ds (tf.data.Dataset): Training data used in training
            - kwargs (dict) :   Used to pipe the arguments into the selected
                            method. This allows methods with different
                            arguments to be called and allows also future
                            implementations/changes.

        Returns:
            list[np.ndarray]: List containing the post activations of each
                                layer and None if the layer is an instance of
                                keras.layers.Flatten
        """

        # get the max m value out of the network
        n_max = max([lay.get_weights()[0].shape[1] for lay in
                     model.layers if isinstance(lay, nngLayer)])

        # create a batch of data out of the training data
        k = 0
        data, res = [], []
        for datapoint, _ in training_ds:
            data.append(datapoint)
            k += datapoint.shape[0]
            if k > n_max:  # add batches until we have enough data
                break
        data = tf.concat(data, axis=0)

        for i, lay in enumerate(model.layers):
            if i == len(model.layers) - 1:
                # last layer should not be changed
                res.append(None)
                break
            data = lay(data)
            if isinstance(lay, nngLayer):
                # 'Due to the dimensionality of the SVD decomposition,
                # evaluating this metric implies a hard constraint of n > Ml'
                res.append(data[:data.shape[1] + 1, :].numpy())
            else:
                res.append(None)

        return res


SUPPORTED_WHERES = {'add_n_with_random_layer': PredefinedWhereWithLayerGrowth,
                    'add_predefined': PredefinedWhere,
                    'autogrow': AutoGrowWhere,
                    'firefly': GradMaxWhere,
                    'trigger': TriggerWhere, }


def get_where_to_grow(*, where_type: str, **kwargs):
    ''' Getter for our supported 'when' methods

    Args.:
        - where_type (str)  :   Selects the strategy.
        - kwargs (dict) :   Used to pipe the arguments into the selected
                            method. This allows methods with different
                            arguments to be called and allows also future
                            implementations/changes.

    Description:
        - initialises and returns selected where-class
        - throws assertion if where_type is not supported.

    '''

    assert where_type.lower() in SUPPORTED_WHERES, \
        f'where_type "{where_type}" is not supported.' \
        f"Supported where_type are:\n {SUPPORTED_WHERES.keys()}"

    return SUPPORTED_WHERES[where_type.lower()](**kwargs)
