import numpy as np
import tensorflow as tf


def set_activation_fn(act_fn):
    """Getter for activation functions 

    Args:
        act_fn (str): Name of the activation function

    Returns:
        function: Activation function

    """
    return tf.keras.activations.get(act_fn)


class nngLayer(tf.keras.layers.Wrapper):
    """Wrapper around keras layer

    Description:
        allows for easier splitting of neurons and to change layers without
        recreating the hole model.

    Methods:
        - add_neurons   :   Adds neurons to the bottom of the layer by
                            recreating the layer in question with altered
                            weights.
        - splitt_neuron :   Splitts neuron in two.
    """

    def __init__(self, *args, activation=None, **kwargs):
        if 'name' not in kwargs:
            if hasattr(args[0], 'name'):
                kwargs['name'] = f"nng_{args[0].name}"
            else:
                kwargs['name'] = "output_layer"
        # when the model build with sequential is called
        # the first time the names of the layers change
        # back to the original ones.
        super().__init__(*args, **kwargs)
        self.activation = set_activation_fn(activation)

    def __call__(self, inputs, *args, **kwargs):

        return self.layer.__call__(inputs, *args, **kwargs)

    def add_neurons(self,
                    n_new: int,
                    add_weights: np.ndarray,
                    add_bias: np.ndarray,
                    in_lay: bool):
        """add_neurons - reinits layer with added weights and biases

        Args:
            n_new (int) :   Number of units that should be added
            add_weights (np.ndarray)    :   weight matrix block corresponding
                                            to the new units
            add_bias (np.ndarray)   :   bias matrix block corresponding
                                        to the new units
            in_lay (bool)   :   if the true bias and weights need to be changed
                                if false only the weights need to be altered.

        Raises:
            NotImplementedError
        """

        old_lay = self.layer

        # Start Grow: ---------------------

        # create new W and b matrices
        add_axs = -1 if in_lay else -2  # matricies are stored transposed

        old_weights = old_lay.get_weights()[0]

        new_shape = list(old_weights.shape)
        new_shape[add_axs] = n_new

        assert add_weights.shape == tuple(
            new_shape), f"{add_weights.shape} vs. {tuple(new_shape)}"

        new_weights = [np.concatenate([old_weights, add_weights],
                                      axis=add_axs)]

        if old_lay.use_bias:
            bias = old_lay.get_weights()[1]

            if in_lay:
                assert len(add_bias) == n_new
                bias = np.concatenate(
                    [bias, add_bias], axis=0)

            new_weights.append(bias)

        # create / generate layer
        common_kwargs = dict()
        for key in ['activation',
                    'use_bias',
                    ]:
            val = getattr(old_lay, key)
            if val is not None:
                common_kwargs[key] = val

        n_new_unit = new_weights[0].shape[-1]

        if isinstance(old_lay, tf.keras.layers.Dense):

            new_lay = tf.keras.layers.Dense(
                n_new_unit,
                weights=new_weights,
                **common_kwargs)
        else:
            raise NotImplementedError(f"Layer type unknown: {old_lay}")

        # End Grow: ---------------------

        # replace layer:
        self.layer = new_lay
        self._name = f"gwn_nng_{self.layer.name}"

    def split_neuron(self, idx_lay, neuron_to_split, mode, in_lay):
        raise NotImplementedError('Splitting')
