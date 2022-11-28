import logging

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from tensorflow import keras

from nng.grow.howtogrow import get_how_to_grow
from nng.grow.layers import nngLayer, set_activation_fn
from nng.grow.models import nng_MLP_Model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def grow_neu_by_adding(*,
                       lay_idx: int,
                       n_neu: int,
                       model,
                       how_type: str,
                       normalized: bool,
                       **kwargs):
    """Grows model by adding neurons to layers.

    Args:
        - lay_idx (int)   :   Layer index (in model.layers) of the layer tobe adjusted.
        - n_neu (int) :   number of neurons to be added to the layer.
        - model (_type_)  :   Network Model to be grown.
        - how_type (str)  :   Type of how the new weights and biases shouldbe initialised.
        - normalized (bool) : if updated layer should be normalized
        - kwargs (dict) :   Used to pipe the arguments into the selectedmethod. This allows methods with differentarguments to be called and allows also futureimplementations/changes.
    """

    how = get_how_to_grow(how_type=how_type,
                          model=model,
                          lay_idx=lay_idx,
                          n_neu=n_neu,
                          **kwargs)
    lays = model.layers

    assert lay_idx < len(lays) - 1, \
        f"The last layers should not be grown."

    lay_in = lays[lay_idx]
    lay_out = lays[lay_idx + 1]

    old_wghts_in = lay_in.get_weights()
    old_wghts_out = lay_out.get_weights()

    n_in, _ = old_wghts_in[0].shape
    _, n_out = old_wghts_out[0].shape

    add_wghts_in, add_bias_in = how.how_to_grow(bck_shape=(n_in, n_neu),
                                                old_wghts=old_wghts_in,
                                                **kwargs)

    add_wghts_out, add_bias_out = how.how_to_grow(bck_shape=(n_neu, n_out),
                                                  old_wghts=old_wghts_out,
                                                  **kwargs)
    if normalized:
        add_wghts_out = tf.keras.utils.normalize(add_wghts_out, axis=-1, order=2)
        add_bias_out = tf.keras.utils.normalize(add_bias_out, axis=-1, order=2)[0]
        add_wghts_in = tf.keras.utils.normalize(add_wghts_in, axis=-1, order=2)
        add_bias_in = tf.keras.utils.normalize(add_bias_in, axis=-1, order=2)[0]

    lay_in.add_neurons(n_neu,
                       add_weights=add_wghts_in,
                       add_bias=add_bias_in,
                       in_lay=True)

    lay_out.add_neurons(n_neu,
                        add_weights=add_wghts_out,
                        add_bias=add_bias_out,
                        in_lay=False)


def adding_it_gm(*,
                 lay_idx: int,
                 n_neu: int,
                 model,
                 loss_function: str,
                 gm_update_mode: str,
                 optimizer: str = 'Adam',
                 hw_lr: float = 0.01,
                 hw_constraint: float | None = None,
                 hw_max_it: int = 75,
                 hw_batch_size: int = 1000,
                 train_data,
                 normalized: bool,
                 **kwargs):
    """Implements an iterative GradMax weights initialisation

    Description:
        - In order to obtain weight matrices which solve equation 4 in
            GradMax (Evci et al.):
                arg max    || IE [ d/dW_l^new L(x) ] ||_F^2
                            + || IE [ d/dW_{l+1}^new L(x) ] ||_F^2
                s.t.: ||W_l^new||_F,||W_l^new||_F <= constraint
            a few steps of projected gradient desent is performed.
        1. For this the weights are first randomly / zero initialised
            according to the `gm_update_mode` see below.
        2. Next the gradients  (d/dW_l^new L(x)) and (d/dW_{l+1}^new L(x))
            are calculated using backpropagation and the objective func is
            evaluated
        3. Only the added new weights are updated using the gradients of
            the objective func.
        4. If the fro-norms of the udated weights exceed the constrained
            region normalizing / clipping is performed.
        5. Steps 2-4 are repeated until hw_max_it is reached.

        - If no constraint is specified the weights of the already existing
            layers are used to compute the constraint see code below.


    Args:
       - lay_idx (int) : Layer index where to add new neurons
       - n_neu (int) : Number of neurons to add
       - model (tf.keras.Model) : Model to modify
       - loss_function (str) : Loss function for training new weights
       - gm_update_mode (str): Describes which of the weight matrixes is set to 0 (in or out)
       - optimizer (str) : Optimizer function for training new weights
       - hw_lr (float): Learning rate
       - hw_constraint (float): Learning constraint
       - hw_max_it (int): Max learning steps
       - hw_batch_size (int) : Amount of data used for training
       - history (tf.keras.callbacks.History): Used to get optimal training time
       - train_data (Dataset) : Dataset used for training new weights
       - normalized: (bool) : If modified layer should be normalized
       - history (tf.keras.callbacks.History) : Used to get optimal training time
       - normalized (bool) : if updated layer should be normalized
       - kwargs (dict) :   Used to pipe the arguments into the selectedmethod. This allows methods with differentarguments to be called and allows also futureimplementations/changes.
    """
    # Init and prepare how method and selection

    tmp_kwargs = kwargs.copy()

    if 'how_type' in tmp_kwargs.keys():
        tmp_kwargs.pop('how_type')

    how_random = get_how_to_grow(how_type='random_baseline', **tmp_kwargs)
    how_const = get_how_to_grow(how_type='const', **tmp_kwargs)

    how = {'in': how_random if gm_update_mode in ['both', 'in']
    else how_const,
           'out': how_random if gm_update_mode in ['both', 'out']
           else how_const, }

    # Get the new weights

    lays = model.layers

    assert lay_idx > 0 and lay_idx < len(lays) - 1, \
        f"The first and last layers should not be grown."

    lay_in = lays[lay_idx]
    lay_out = lays[lay_idx + 1]

    old_wghts_in = lay_in.get_weights()
    old_wghts_out = lay_out.get_weights()

    n_in, _ = old_wghts_in[0].shape
    _, n_out = old_wghts_out[0].shape

    add_wghts_in, add_bias_in = how['in'].how_to_grow(
        bck_shape=(n_in, n_neu),
        old_wghts=old_wghts_in,  # If 'in' or 'both'
        const=0,  # if 'out'
        **kwargs)

    add_wghts_out, add_bias_out = how['out'].how_to_grow(
        bck_shape=(n_neu, n_out),
        old_wghts=old_wghts_in,  # needed if 'out' or 'both'
        const=0,  # needed if 'in'
        **kwargs)

    # Add new weights to layers

    lay_in.add_neurons(n_neu,
                       add_weights=add_wghts_in,
                       add_bias=add_bias_in,
                       in_lay=True)

    lay_out.add_neurons(n_neu,
                        add_weights=add_wghts_out,
                        add_bias=add_bias_out,
                        in_lay=False)

    # tf function for the optimization of the weights

    @tf.function
    def gm_it_step(*, data,
                   targets,
                   lay_idx,
                   n_neu,
                   loss_fn,
                   model,
                   gm_update_mode="both"):

        assert gm_update_mode.lower() in ['both', 'in', 'out'], \
            f"{gm_update_mode = } not supported. " \
            f"Supported are {['both', 'in', 'out']}"

        x = data
        with tf.GradientTape(persistent=True) as tape:
            # only the grads up from lay_idx are needed
            with tape.stop_recording():
                for lay in model.layers[:lay_idx]:
                    x = lay(x)

            for lay in model.layers[lay_idx:]:
                x = lay(x)

            # compute the value of our custom objective function
            tmp_loss = loss_fn(targets, x)

            grad_in_lay = tape.gradient(tmp_loss,
                                        model.layers[lay_idx].weights[0])
            grad_out_lay = tape.gradient(tmp_loss,
                                         model.layers[lay_idx + 1].weights[0])

            # we are only interested in the grads wrt the new neurons
            tmp_in_grad = grad_in_lay[..., n_neu:]
            tmp_out_grad = grad_out_lay[n_neu:, ...]

            # loss function in Eq. 4 in GradMax Evci et al.
            # NOTE the frobenius norm of a matrix is equivalent to the
            # euclidean norm of the flatten matrix. And that we want to
            # maximize (=> -)
            loss = -tf.norm(tmp_in_grad, ord='euclidean') ** 2 \
                   - tf.norm(tmp_out_grad, ord='euclidean') ** 2

        if gm_update_mode in ['both', 'in']:
            # Grads for W^{new}_{\ell}
            tmp_in_grad2 = tape.gradient(
                loss, model.layers[lay_idx].weights[0])
            # mask the gradients
            fin_in_grad = tf.concat((tf.zeros_like(tmp_in_grad2[..., :-n_neu]),
                                     tmp_in_grad2[..., -n_neu:]), axis=1)

            optimizer.apply_gradients(
                [(fin_in_grad, model.layers[lay_idx].weights[0])])

        if gm_update_mode in ['both', 'out']:
            # Grads for W^{new}_{\ell+1}
            tmp_out_grad2 = tape.gradient(
                loss, model.layers[lay_idx + 1].weights[0])
            fin_out_grad = tf.concat(
                (tf.zeros_like(tmp_out_grad2[:-n_neu, ...]),
                 tmp_out_grad2[-n_neu:, ...]), axis=0)

            optimizer.apply_gradients(
                [(fin_out_grad, model.layers[lay_idx + 1].weights[0])])

        del tape

        return loss

    # Optimizing the random weights wrt. to the objective in Eq. 4 GradMax

    optimizer = getattr(keras.optimizers, optimizer)(learning_rate=hw_lr)

    loss_fn = getattr(keras.losses, loss_function)(from_logits=True)

    # -----------------------

    k = 0
    batch_data = []
    batch_targets = []

    for data, targets in train_data:
        batch_data.append(data)
        batch_targets.append(targets)
        k += len(data)
        if k >= hw_batch_size:
            break

    batch = tf.concat(batch_data, axis=0)
    targets = tf.concat(batch_targets, axis=0)

    # -----------------

    # Use mean of the euclidead norm of the weights of old_in old_out
    if hw_constraint is None:
        constraint = n_neu * ((old_wghts_in[0] ** 2).mean(axis=1).sum()
                              + (old_wghts_out[0] ** 2).mean(axis=0).sum()) / 2
        logging.info(constraint)
    elif isinstance(hw_constraint, float):
        assert hw_constraint > 0, \
            f"{hw_constraint=} needs to be positive to be meaningful."
        constraint = hw_constraint
    else:
        raise ValueError(f"{hw_constraint=} needs to be a non negative "
                         "float or none.")

    logging.info("\n\n [IT_GRADMAX]" + '.' * 120 + "\n\n")
    logging.info("             Starting projected gradient descent...")

    for _i in range(hw_max_it):

        # gradient step

        obj_val = gm_it_step(data=batch, targets=targets, lay_idx=lay_idx,
                             n_neu=n_neu, gm_update_mode=gm_update_mode,
                             model=model, loss_fn=loss_fn)

        nrm = None

        # projection step

        if gm_update_mode in ['both', 'in']:
            lay = model.layers[lay_idx]
            lay_w, _ = lay.get_weights()

            nrm = np.linalg.norm(lay_w[..., -n_neu:], ord='fro')
            if nrm > constraint:
                lay_w[..., -n_neu:] *= constraint / nrm
            lay.weights[0].assign(lay_w)

        nrm1 = nrm.copy() if nrm else None

        if gm_update_mode in ['both', 'out']:
            lay = model.layers[lay_idx + 1]
            lay_w, _ = lay.get_weights()

            nrm = np.linalg.norm(lay_w[-n_neu:, ...], ord='fro')
            if nrm > constraint:
                lay_w[-n_neu:, ...] *= constraint / nrm
            lay.weights[0].assign(lay_w)

        if normalized:
            model.layers[lay_idx].weights[0] = tf.keras.utils.normalize(model.layers[lay_idx].weights[0], axis=-1,
                                                                        order=2)
            model.layers[lay_idx + 1].weights[0] = tf.keras.utils.normalize(model.layers[lay_idx + 1].weights[0],
                                                                            axis=-1, order=2)
            model.layers[lay_idx].weights[1] = \
                tf.keras.utils.normalize(model.layers[lay_idx].weights[1], axis=-1, order=2)[0]
            model.layers[lay_idx + 1].weights[1] = \
                tf.keras.utils.normalize(model.layers[lay_idx + 1].weights[1], axis=-1, order=2)[0]
        logging.info(
            f"{_i}/{hw_max_it}: {-obj_val.numpy() = } "
            f"| nrms: in:{nrm1} , out:{nrm}")

    logging.info("\n\n" + "." * 120)


def adding_auto_lite(*,
                     lay_idx: int,
                     n_neu: int,
                     model,
                     loss_function: str,
                     optimizer: str = 'Adam',
                     hw_lr: float = 0.01,
                     hw_constraint: float | None = None,
                     hw_batch_size: int = 1_000,
                     hw_max_it: int = 10,
                     history,
                     train_data,
                     normalized: bool,
                     **kwargs):
    """Implements weight update inspired by the Adam init from AutoGrow
        Wen, Wei, et al. "Autogrow: Automatic layer growing in deep
        convolutional networks." Proceedings of the 26th ACM SIGKDD
        International Conference on Knowledge Discovery & Data Mining. 2020.

    Description:
    - For this the weights of the `lay_idx` and the `lay_idx +1` layer
        are changed.
    - The random_baseline is used to initialize the weights.
    - The AdamInit approach is training only the new weights which are added.
      The training of the new weights is only done max 10 epochs, or until
      the training loss is <= as the training loss before adding the weights.


    Args:
       - lay_idx (int) : Layer index where to add new neurons
       - n_neu (int) : Number of neurons to add
       - model (tf.keras.Model) : Model to modify
       - loss_function (str) : Loss function for training new weights
       - optimizer (str) : Optimizer function for training new weights
       - hw_lr (float): Learning rate
       - hw_constraint (float): Learning constraint
       - hw_max_it (int): Max learning steps
       - hw_batch_size (int) : Amount of data used for training
       - history (tf.keras.callbacks.History): Used to get optimal training time
       - train_data (Dataset) : Dataset used for training new weights
       - normalized: (bool) : If modified layer should be normalized
       - history (tf.keras.callbacks.History) : Used to get optimal training time
       - normalized (bool) : if updated layer should be normalized
       - kwargs (dict) :   Used to pipe the arguments into the selectedmethod. This allows methods with differentarguments to be called and allows also futureimplementations/changes.
    """
    counter = 0
    # Init and prepare how method and selection
    tmp_kwargs = kwargs.copy()

    if 'how_type' in tmp_kwargs.keys():
        tmp_kwargs.pop('how_type')

    how = get_how_to_grow(how_type='random_baseline', **tmp_kwargs)

    # Get the new weights

    lays = model.layers

    assert lay_idx > 0 and lay_idx < len(lays) - 1, \
        f"The first and last layers should not be grown."

    lay_in = lays[lay_idx]
    lay_out = lays[lay_idx + 1]

    old_wghts_in = lay_in.get_weights()
    old_wghts_out = lay_out.get_weights()

    n_in, _ = old_wghts_in[0].shape
    _, n_out = old_wghts_out[0].shape

    add_wghts_in, add_bias_in = how.how_to_grow(
        bck_shape=(n_in, n_neu),
        old_wghts=old_wghts_in,  # If 'in' or 'both'
        **kwargs)

    add_wghts_out, add_bias_out = how.how_to_grow(
        bck_shape=(n_neu, n_out),
        old_wghts=old_wghts_in,  # needed if 'out' or 'both'
        **kwargs)

    # Add new weights to layers

    lay_in.add_neurons(n_neu,
                       add_weights=add_wghts_in,
                       add_bias=add_bias_in,
                       in_lay=True)

    lay_out.add_neurons(n_neu,
                        add_weights=add_wghts_out,
                        add_bias=add_bias_out,
                        in_lay=False)

    # tf function for the optimization of the weights
    @tf.function
    def auto_it_step(*, data,
                     targets,
                     lay_idx,
                     loss_fn,
                     model):

        x = data
        with tf.GradientTape(persistent=True) as tape:

            # only the grads up from lay_idx are needed
            with tape.stop_recording():
                for lay in model.layers[:lay_idx]:
                    x = lay(x)

            for lay in model.layers[lay_idx:]:
                x = lay(x)

            # compute the loss with the added weights
            loss = loss_fn(targets, x)

        # Grads for W^{new}_{\ell}

        grad_in_lay = tape.gradient(loss, model.layers[lay_idx].weights[0])
        optimizer.apply_gradients(
            [(grad_in_lay, model.layers[lay_idx].weights[0])])

        # Grads for W^{new}_{\ell+1}
        grad_out_lay = tape.gradient(loss, model.layers[lay_idx + 1].weights[0])
        optimizer.apply_gradients(
            [(grad_out_lay, model.layers[lay_idx + 1].weights[0])])

        del tape

        return loss

    # Optimizing the random weights wrt. to the objective in Eq. 4 AutoGrow

    optimizer = getattr(keras.optimizers, optimizer)(learning_rate=hw_lr)
    loss_fn = getattr(keras.losses, loss_function)(from_logits=True)

    # -----------------------

    k = 0
    batch_data = []
    batch_targets = []

    for data, targets in train_data:
        batch_data.append(data)
        batch_targets.append(targets)
        k += len(data)
        if k >= hw_batch_size:
            break

    batch = tf.concat(batch_data, axis=0)
    targets = tf.concat(batch_targets, axis=0)

    # -----------------

    def breaking_condition(loss, counter, max_count):
        if counter >= max_count:
            return True
        elif history.history['loss'][-1] >= loss.numpy() / 100:
            return True
        return False

    # Iteratively update the new weights in a greedy way:

    for i in range(hw_max_it):

        # gradient step

        loss = auto_it_step(data=batch, targets=targets, lay_idx=lay_idx,
                            model=model, loss_fn=loss_fn)
        logging.info(f"{i}/{hw_max_it}: {loss.numpy() = } ")
        counter += 1
        if breaking_condition(loss, counter, hw_max_it):
            if normalized:
                model.layers[lay_idx].weights[0] = tf.keras.utils.normalize(model.layers[lay_idx].weights[0], axis=-1,
                                                                            order=2)
                model.layers[lay_idx + 1].weights[0] = tf.keras.utils.normalize(model.layers[lay_idx + 1].weights[0],
                                                                                axis=-1, order=2)
                model.layers[lay_idx].weights[1] = \
                    tf.keras.utils.normalize(model.layers[lay_idx].weights[1], axis=-1, order=2)[0]
                model.layers[lay_idx + 1].weights[1] = \
                    tf.keras.utils.normalize(model.layers[lay_idx + 1].weights[1], axis=-1, order=2)[0]
            break


def grow_neu_by_splitting(**kwargs):
    """Grows Network by splitting

    Raises:
        NotImplementedError: _description_
    """
    raise NotImplementedError


def grow_lay_baseline(*, model,
                      lay_idx: int,
                      n_neu: int,
                      how_type: str,
                      activation_function: str,
                      data_shape: tuple[int],
                      normalized: bool,
                      **kwargs):
    """Grows Layer to the model by just adding specified number of neurons.

    Args:
        - model (tf.keras.Model)    : Model to be trained
        - lay_idx (int)   :   Index where the layer should be inserted. Thelayer that has currently the index lay_idxwill have the index (lay_idx +1) after growing.
        - n_neu (int) :   Number of units in the new layer. NOTE: Thisnumber must be greater or equal to the number ofneurons in the current layer with index lay_idx.
        - how_type (str)  :   Type of how the new weights and biases shouldbe initialised.
        - activation_function (str)   :   Activation function for the newlayer.
        - data_shape (tuple[int]) :   Dimensions of the Input that is feedto the first layer of the network. This is needed to init the network after it was recreated.
        - normalized (bool) : if updated layer should be normalized
        - kwargs (dict) :   Used to pipe the arguments into the selectedmethod. This allows methods with differentarguments to be called and allows also futureimplementations/changes.

    Returns:
        new_model   :   Model with added layer. Apart form that it is the
                        same as before.
    """
    assert isinstance(model, nng_MLP_Model), \
        f"Currently only nng_MLP_MODELs are supported."

    how = get_how_to_grow(how_type=how_type)
    old_lays = model.layers

    # Output layer has no (or special activation) hence it should not be
    # altered.
    assert lay_idx < len(old_lays) and lay_idx > 0, \
        "Output and Flatten layer (idx 0) are fixed " \
        f'(1 <= lay_idx < {len(model.layers)}).' \
        f' Given: {lay_idx = }.'

    nxt_lay = old_lays[lay_idx]
    nxt_lay_wghts = nxt_lay.get_weights()

    try:
        n_in, n_out = nxt_lay_wghts[0].shape
    except IndexError:
        raise IndexError('Probably the layer has not been build.')

    assert n_neu >= n_in, f"{n_neu = }, {n_in = }" \
                          "\nadd_neuron is not designed for pruning. " \
                          "If the layer should contain a smaller number of neurons than " \
                          "the one before it create the layer afterwards from scratch " \
                          "as a new nngLayer (some weights might be re-used form the old one."

    # generate new layers
    new_wghts = how.how_to_grow(bck_shape=(
        n_in, n_neu), old_wghts=None, **kwargs)

    actv_fn = set_activation_fn(activation_function)
    new_lay = nngLayer(Dense(n_neu,
                             weights=new_wghts,
                             activation=actv_fn,
                             ))

    new_lay._name = f"new_{new_lay.name}"

    # adapt layer after the new layer in order to process with the new units
    n_nxt_wghts, _ = how.how_to_grow(bck_shape=((n_neu - n_in), n_out),
                                     old_wghts=nxt_lay_wghts)

    if normalized:
        n_nxt_wghts = tf.keras.utils.normalize(n_nxt_wghts, axis=-1, order=2)

    nxt_lay.add_neurons((n_neu - n_in),
                        add_weights=n_nxt_wghts,
                        add_bias=None, in_lay=False)

    # prepare initialisation of a model copy
    first_lay = old_lays[0]
    hidd_lays = old_lays[1:lay_idx]
    hidd_lays.append(new_lay)
    hidd_lays += old_lays[lay_idx:]
    last_lay = hidd_lays.pop()

    new_model = nng_MLP_Model(first_lay=first_lay,
                              layers=hidd_lays,
                              last_lay=last_lay)

    # THIS STEP IS IMPORTANT
    # such that every layer is initialised and can be grown
    new_model(keras.Input(data_shape))

    return new_model


def adding_firefly(*,
                   lay_idx: int,
                   n_neu: int,
                   model,
                   loss_function: str,
                   optimizer: str = 'Adam',
                   hw_lr: float = 0.01,
                   hw_max_it: int = 100,
                   train_data,
                   normalized,
                   **kwargs):
    """Grows model by adding neurons to layers with firefly like meyhod.

    It's not exactly firefly method with which the existing neurons are split
    and the new neurons are then added.
    In this function, we only consider adding new neurons.
    In the code of GradMax, the magnitude(epsilon) is only gotten once by the
    mean value of initial random value.
    However, in the paper of Firefly, the optimization of magnitude(epsilon)
    and direction(delta) should be interleaved, and for optimization of the
    magnitude(epsilon), the value chosen of magnitude(epsilon) should be the
    largest one instead of the mean value of the whole.(This is a bit like
    "maximization expectation" mindset.)
    In this function below,  we correct this point.
    Args:
        - lay_idx (int)   :   Layer index (in model.layers) of the layer tobe
                            adjusted.
        - n_neu (int) :   number of neurons to be added to the layer.
        - model (tf.keras.Model)  :   Network Model to be grown.
        - loss_function(str)  : type of loss function.
        - optimizer (str)  :   type of optimizer.
        - hw_lr (float)   :   learning rate for gradient descent of new weights.
        - hw_max_it (int) :   number of epochs for gradient descent of new
                            weights.
        - train_data (PrefechedDataset)   :   training data for gradient descent
                                            of new weights.
        - normalized (bool) : if updated layer should be normalized
        - kwargs (dict) :   Used to pipe the arguments into the selectedmethod. This allows methods with differentarguments to be called and allows also futureimplementations/changes.
    """
    optimizer = getattr(keras.optimizers, optimizer)(learning_rate=hw_lr)
    loss_function = getattr(keras.losses, loss_function)(from_logits=True)
    # initialize:add neuron using random method
    tmp_kwargs = kwargs.copy()

    if 'how_type' in tmp_kwargs.keys():
        tmp_kwargs.pop('how_type')

    how = get_how_to_grow(how_type='random_baseline', **tmp_kwargs)
    lays = model.layers

    assert lay_idx > 0 and lay_idx < len(lays) - 1, \
        f"The first and last layers should not be grown."

    lay_in = lays[lay_idx]
    lay_out = lays[lay_idx + 1]

    old_wghts_in = lay_in.get_weights()
    old_wghts_out = lay_out.get_weights()

    n_in, n_old_neuron = old_wghts_in[0].shape
    _, n_out = old_wghts_out[0].shape

    add_wghts_in, add_bias_in = how.how_to_grow(bck_shape=(n_in, n_neu),
                                                old_wghts=old_wghts_in,
                                                **kwargs)

    add_wghts_out, add_bias_out = how.how_to_grow(bck_shape=(n_neu, n_out),
                                                  old_wghts=old_wghts_out,
                                                  **kwargs)

    max_magnitudes = []

    # initialize magnitude with the max value of random initialization
    normalized_add_weights_in = tf.keras.utils.normalize(
        add_wghts_in, axis=-1, order=2)
    normalized_add_weights_out = tf.keras.utils.normalize(
        add_wghts_out, axis=-2, order=2)

    max_magnitudes.append(np.max(normalized_add_weights_in))
    max_magnitudes.append(np.max(normalized_add_weights_out))

    lay_in.add_neurons(n_neu,
                       add_weights=normalized_add_weights_in,
                       add_bias=add_bias_in,
                       in_lay=True)

    lay_out.add_neurons(n_neu,
                        add_weights=normalized_add_weights_out,
                        add_bias=add_bias_out,
                        in_lay=False)

    batch, targets = train_data.as_numpy_iterator().next()

    @tf.function
    def tuning(inputs, targets):
        with tf.GradientTape(persistent=True) as tape:
            loss = loss_function(targets, model(inputs))
        grads_in = tape.gradient(loss, model.layers[lay_idx].weights[0])
        grad_out = tape.gradient(loss, model.layers[lay_idx + 1].weights[0])
        grads = [grads_in, grad_out]
        final_grads = []
        for concat_axis, grad in zip([-1, -2], grads):
            # Apply gradient only on new weights, zero out the rest.
            old_wgrad, new_wgrad = tf.split(
                grad, [n_old_neuron, -1], axis=concat_axis)
            final_grad = tf.concat(
                [tf.zeros_like(old_wgrad), new_wgrad], axis=concat_axis)
            final_grads.append(final_grad)
        # Update the new weights of the model.
        optimizer.apply_gradients(
            [(final_grads[0], model.layers[lay_idx].weights[0])])
        optimizer.apply_gradients(
            [(final_grads[1], model.layers[lay_idx + 1].weights[0])])
        return loss

    for i in range(hw_max_it):
        loss = tuning(batch, targets)

        logging.info('In transformer: iter: %d, loss: %s', i, loss)
        # optimize delta
        weights = [model.layers[lay_idx].weights[0],
                   model.layers[lay_idx + 1].weights[0]]
        for concat_axis, weight, max_magnitude in zip([-1, -2], weights,
                                                      max_magnitudes):
            old_w, new_w = tf.split(
                weight, [n_old_neuron, -1], axis=concat_axis)
            norm_weights__mag = tf.keras.utils.normalize(
                new_w, axis=concat_axis, order=2) * max_magnitude
            weight.assign(
                tf.concat([old_w, norm_weights__mag], axis=concat_axis))
            # optimize epsilon
            if concat_axis == -1:
                max_magnitudes[0] = np.max(norm_weights__mag)
            elif concat_axis == -2:
                max_magnitudes[1] = np.max(norm_weights__mag)

            if normalized and hw_max_it == i:
                model.layers[lay_idx].weights[0] = tf.keras.utils.normalize(model.layers[lay_idx].weights[0], axis=-1,
                                                                            order=2)
                model.layers[lay_idx + 1].weights[0] = tf.keras.utils.normalize(model.layers[lay_idx + 1].weights[0],
                                                                                axis=-1, order=2)
                model.layers[lay_idx].weights[1] = \
                    tf.keras.utils.normalize(model.layers[lay_idx].weights[1], axis=-1, order=2)[0]
                model.layers[lay_idx + 1].weights[1] = \
                    tf.keras.utils.normalize(model.layers[lay_idx + 1].weights[1], axis=-1, order=2)[0]

    logging.info('In transformer: final loss: %s', loss.numpy())


def grow_lay_idempotent(*, model: keras.Model,
                        lay_idx: int,
                        n_neu: int,
                        how_type: str,
                        activation_function: str,
                        data_shape: tuple[int],
                        **kwargs):
    """Grows Layer to the model by idempotent layer growth.

    Args:
        - model (tf.keras.Model)    :   Network to be grown.
        - lay_idx (int)   :   Index where the layer should be inserted. The
                            layer that has currently the index lay_idx
                            will have the index (lay_idx +1) after growing.
        - n_neu (int) :   Number of units in the new layer. NOTE: This
                        number must be greater or equal to the number of
                        neurons in the current layer with index lay_idx.
        - how_type (str)  :   Type of how the new weights and biases should
                            be initialised.
        - activation_function (str)   :   Activation function for the new
                                        layer.
        - data_shape (tuple[int]) :   Dimensions of the Input that is feed
                                    to the first layer of the network.
                                    This is needed to init the network
                                    after it was recreated.
        - kwargs (dict) :   Used to pipe the arguments into the selectedmethod. This allows methods with differentarguments to be called and allows also futureimplementations/changes.

    Returns:
        new_model   :   Model with added layer. Apart form that it is the
                        same as before.
    """
    assert isinstance(model, nng_MLP_Model), \
        f"Currently only nng_MLP_MODELs are supported."

    how = get_how_to_grow(how_type=how_type)
    old_lays = model.layers

    # Output layer has no (or special activation) hence it should not be
    # altered.
    assert lay_idx < len(old_lays) and lay_idx > 0, \
        "Output and Flatten layer (idx 0) are fixed " \
        f'(1 <= lay_idx < {len(model.layers)}).' \
        f' Given: {lay_idx = }.'

    nxt_lay = old_lays[lay_idx]
    nxt_lay_wghts = nxt_lay.get_weights()

    try:
        n_in, n_out = nxt_lay_wghts[0].shape
    except IndexError:
        raise IndexError('Probably the layer has not been build.')

    assert n_neu >= n_in, f"{n_neu = }, {n_in = }" \
                          "\nadd_neuron is not designed for pruning. " \
                          "If the layer should contain a smaller number of neurons than " \
                          "the one before it create the layer afterwards from scratch " \
                          "as a new nngLayer (some weights might be re-used form the old one."

    # generate new layers
    new_wghts = how.how_to_grow(bck_shape=(
        n_in, n_neu), old_wghts=None, **kwargs)

    new_wghts[0] = np.eye(n_in, dtype=float)

    actv_fn = set_activation_fn(activation_function)
    new_lay = nngLayer(Dense(n_neu,
                             weights=new_wghts,
                             activation=actv_fn,
                             ))

    new_lay._name = f"new_{new_lay.layer.name}"

    # adapt layer after the new layer in order to process with the new units
    n_nxt_wghts, _ = how.how_to_grow(bck_shape=((n_neu - n_in), n_out),
                                     old_wghts=nxt_lay_wghts)
    nxt_lay.add_neurons((n_neu - n_in),
                        add_weights=n_nxt_wghts,
                        add_bias=None, in_lay=False)

    # prepare initialisation of a model copy
    first_lay = old_lays[0]
    hidd_lays = old_lays[1:lay_idx]
    hidd_lays.append(new_lay)
    hidd_lays += old_lays[lay_idx:]
    last_lay = hidd_lays.pop()

    new_model = nng_MLP_Model(first_lay=first_lay,
                              layers=hidd_lays,
                              last_lay=last_lay)

    # THIS STEP IS IMPORTANT
    # such that every layer is initialised and can be grown
    new_model(keras.Input(data_shape))

    return new_model


NEURON_GROW_TYPES = {
    'adding': grow_neu_by_adding,
    'splitting': grow_neu_by_splitting,
    'adding_it_gradmax': adding_it_gm,
    'adding_auto_lite': adding_auto_lite,
    'adding_firefly': adding_firefly,
}

LAYER_GROW_TYPES = {
    'baseline': grow_lay_baseline,
    'idempotent': grow_lay_idempotent,
    # 'res_lay' : ,
    # 'p_arametric': ,
}


class ModelTransformer:
    """Implements the model transformation method.

    Args:
        - old_model (tf.keras.Model)    :   Old model.
        - layer_to_modify (int) :   The layer which will grow.
        - loss_function (str)   :   Defines the loss function to be used for
                                    training the model.
        - learning_step (float) :   Defines the training steps to be used for
                                    training the model.
        - optimiser (str)   :   Defines the optimiser to be used for training
                                the model.
        - n_neurons (int) :   Number of new neurons to grow on the layer
        - metric (str)  :   Defines the metrics to be used for training the
                            model.
    """

    def __init__(self,
                 optimiser='Adam',
                 learning_step=0.001,
                 loss_function='SparseCategoricalCrossentropy',
                 metric='SparseCategoricalAccuracy'):
        self.optimiser = getattr(tf.keras.optimizers, optimiser)
        self.learning_step = learning_step
        self.loss_function = getattr(tf.keras.losses, loss_function)
        self.metric = getattr(tf.keras.metrics, metric)

    def grow_neuron(self, *, grow_neu_type: str, **kwargs):
        """Interface to the growing methods for neurons

        Args:
            grow_neu_type (str): switch
        """

        assert grow_neu_type.lower() in NEURON_GROW_TYPES, \
            f"{grow_neu_type.lower()} is not a supported grow type." \
            f"Supported grow types are: \n {NEURON_GROW_TYPES.keys()}"

        NEURON_GROW_TYPES[grow_neu_type.lower()](**kwargs)

    def grow_layer(self, *, grow_lay_type: str, **kwargs):
        """Interface to the growing methods for layers

        Args:
            grow_lay_type (str): switch

        Returns:
            new_model : model with grown layer.
        """

        assert grow_lay_type.lower() in LAYER_GROW_TYPES, \
            f"{grow_lay_type.lower()} is not a supported grow type." \
            f"Supported grow types are: \n {LAYER_GROW_TYPES.keys()}"

        return LAYER_GROW_TYPES[grow_lay_type.lower()](**kwargs)
