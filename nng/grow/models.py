import logging

from keras.layers import Dense, Flatten
from tensorflow import keras

from nng.grow.layers import nngLayer

logger = logging.getLogger(__name__)


class Model:
    """Implements the basic model class

    Args:
        model (tf.keras.Model): Model to train.
    """

    def __init__(self, model):
        self.model = model

    def get_model(self):
        """Function that returns the model.

        Returns:
            - model (tf.keras.Model): Model to train.
        """
        return self.model

    def set_model(self, model):
        """Function that sets the model within the class.

        """
        self.model = model


class nng_MLP_Model(keras.Model):
    """
    Model class for growing Multilayer Perceptrons
    """

    def __init__(self, first_lay, layers, last_lay):
        """Init function of the nng_MLP_Model

        """
        super().__init__()

        if isinstance(first_lay, nngLayer):
            raise TypeError('First layer is not nng!')
        else:
            self.first_lay = first_lay
        self.lays = []
        for lay in layers:
            self.lays.append(lay)
        self.last_lay = last_lay

    def call(self, inp):
        x = self.first_lay(inp)
        for lay in self.lays:
            x = lay(x)
        outp = self.last_lay(x)
        return outp

    def __repr__(self):
        rep = f'---- {self.name} -----------------------------------\n'
        for lay in self.layers:
            rep += str(lay.name)
            rep += '\t\t\t'
            try:
                ind, outd = lay.get_weights()[0].shape
            except IndexError:
                ind, outd = "-", "-"
            rep += f"[in, out] = [{ind}, {outd}]\n"
        rep += '-------------------------------------------------------'
        return rep


class nng_MLP_Student(keras.Model):
    """
    Model class for growing Multilayer Perceptrons
    """

    def __init__(self, layers, last_lay):
        """ Init function of the nng_MLP_Student

        """
        super().__init__()

        self.lays = []
        for lay in layers:
            self.lays.append(lay)
        self.last_lay = last_lay

    def call(self, x):
        for lay in self.lays:
            x = lay(x)
        outp = self.last_lay(x)
        return outp

    def __repr__(self):
        rep = f'---- {self.name} -----------------------------------\n'
        for lay in self.layers:
            rep += str(lay.name)
            rep += '\t\t\t'
            try:
                ind, outd = lay.get_weights()[0].shape
            except IndexError:
                ind, outd = "-", "-"
            rep += f"[in, out] = [{ind}, {outd}]\n"
        rep += '-------------------------------------------------------'
        return rep


def create_nng_mlp(*, out_dim,
                   in_dim=None,
                   arch: list[int] = [500, 400],
                   activation: str = 'relu'):
    '''
    Args:
        - aut
        - arch  :   specifies the hidden layer eg. not the input
                    dim or the output dim
    '''
    first_lay = Flatten()

    hidd_lays = list()
    for n in arch:
        hidd_lays.append(
            nngLayer(
                Dense(n, activation=activation)
            ))
    last_lay = nngLayer(Dense(out_dim))

    return nng_MLP_Model(first_lay=first_lay,
                         layers=hidd_lays,
                         last_lay=last_lay)


def create_nng_student_mlp(*,
                           m_i: int,
                           m_h: list[int],
                           m_o: int,
                           kernel_init,
                           bias_init,
                           activation: str = 'relu'
                           ):
    '''
    Args:
        - m_i (int)
        - m_o (m_o)
        - kernel_init (keras.initializers)
        - bias_init (keras.initializers)
        - activation (str)  :  Activation function
    '''

    h_lays = list()
    for n in m_h:
        h_lays.append(nngLayer(Dense(n,
                                     activation=activation,
                                     kernel_initializer=kernel_init,
                                     bias_initializer=bias_init)))
    out_lay = nngLayer(Dense(m_o,
                             kernel_initializer=kernel_init,
                             bias_initializer=bias_init))

    return nng_MLP_Student(layers=h_lays,
                           last_lay=out_lay)


SUPPORTED_MODELS = {
    'MLP': create_nng_mlp,
    'STUDENT_MLP': create_nng_student_mlp}


def create_nng(**kwargs):
    """Getter for the models

    Description:
        Abstracts nng selection from the pipeline into this file and the
        config.

    Args:
        - kwargs (dict) :   Used to pipe the arguments into the selected
            method. This allows methods with different
            arguments to be called and allows also future
            implementations/changes.

    Returns:
        - nng model
    """
    model_type = kwargs.pop('model_type')

    assert model_type.upper() in SUPPORTED_MODELS, \
        f"{model_type=} not supported. Supported are: {SUPPORTED_MODELS.keys()}"

    return (SUPPORTED_MODELS[model_type.upper()](**kwargs))
