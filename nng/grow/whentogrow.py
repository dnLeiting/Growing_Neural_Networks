import logging

logger = logging.getLogger(__name__)


class WhenToGrow:
    """Implements the when to grow methods.

    Attr:
        - when_typel (tf.keras.when_typel)    :   Machine Learning when_typel.
        - max_training (int)    :   Max amount of training epochs before
                                    growing.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def when_to_grow(self, **kwargs):
        raise NotImplementedError()


class AutoGrowWhen(WhenToGrow):

    def when_to_grow(self, *, history, flops_history, **kwargs) -> bool:
        """Convergent Growth methoed defined in the Autogrow paper

        Args:
            - history (list of tf.keras.callbacks.History)  :
                                    List of previous training history.
            - flops_history (list of int): Flops of the last training  loops
            - kwargs (dict) :   Used to pipe the arguments into the selectedmethod.
                                    This allows methods with differentarguments to be
                                    called and allows also futureimplementations/changes.

        Returns:
            (bool)  :   Bool if the when_typel should be grow.
        """
        val_accuracy_difference = \
            history[-1].history['val_sparse_categorical_accuracy'][0] \
            - history[-1].history['val_sparse_categorical_accuracy'][-1]

        if self.kwargs['val_sparse_categorical_accuracy'] \
                > val_accuracy_difference:
            return True

        return False


class PredefinedWhen(WhenToGrow):

    def when_to_grow(self, *, history, flops_history, **kwargs) -> bool:
        """Default function for deciding when to grow.

        Args:
            - history (list of tf.keras.callbacks.History)  :
                                    List of previous training history.
            - flops_history (list of int): Flops of the last training  loops
            - kwargs (dict) :   Used to pipe the arguments into the selectedmethod.
                                    This allows methods with differentarguments to be
                                    called and allows also futureimplementations/changes.

        Returns:
            (bool)  :   Bool if the when_typel should be grow.
        """
        number_of_epochs = 0

        for current_history in history:

            number_of_epochs = len(current_history.history['loss']) \
                               + number_of_epochs

            if number_of_epochs >= self.kwargs['max_training']:
                return True

        return False


class PredefinedWhenValidationLossFlops(WhenToGrow):

    def when_to_grow(self, *, history, flops_history, **kwargs) -> bool:
        """Validation loss and FLOPS based function for deciding when to grow

        Args:
            - history (list of tf.keras.callbacks.History)  :
                                    List of previous training history.
            - flops_history (list of int): Flops of the last training  loops
            - kwargs (dict) :   Used to pipe the arguments into the selectedmethod.
                                    This allows methods with differentarguments to be
                                    called and allows also futureimplementations/changes.

        Returns:
            (bool)  :   Bool if the when_typel should be grow.
        """
        val_loss_difference = history[-1].history['val_loss'][0] \
                              - history[-1].history['val_loss'][-1]

        logger.info('val_loss_difference: ' + str(val_loss_difference))

        val_loss_difference_percentage = val_loss_difference \
                                         / history[-1].history['val_loss'][0]

        average_flops = sum(flops_history) / len(flops_history)
        latest_flops_weighted = flops_history[-1] / average_flops

        val_loss_difference_percentage_weighted_flops = \
            val_loss_difference_percentage * latest_flops_weighted

        # This is letting the network grow in the early stages of the nng
        logger.info('val_loss_difference_percentage_weighted_flops: '
                    + str(val_loss_difference_percentage_weighted_flops))

        if val_loss_difference_percentage_weighted_flops \
                < self.kwargs['growing_threshold_flops']:
            return True

        return False


class PredefinedWhenValidationLossEpoch(WhenToGrow):

    def when_to_grow(self, *, history, flops_history, **kwargs) -> bool:
        """Validation loss and Epochs based function for deciding when to grow

        Args:
            - history (list of tf.keras.callbacks.History)  :
                                    List of previous training history.
            - flops_history (list of int): Flops of the last training  loops
            - kwargs (dict) :   Used to pipe the arguments into the selectedmethod.
                                    This allows methods with differentarguments to be
                                    called and allows also futureimplementations/changes.

        Returns:
            (bool)  :   Bool if the when_typel should be grow.
        """
        val_loss_difference = history[-1].history['val_loss'][0] - \
                              history[-1].history['val_loss'][-1]

        logger.info('val_loss_difference: ' + str(val_loss_difference))

        val_loss_difference_percentage = val_loss_difference \
                                         / history[-1].history['val_loss'][0]

        val_loss_difference_percentage_epochs = val_loss_difference_percentage \
                                                / len(history[-1].history['val_loss'])

        logger.info('val_loss_difference_percentage_epochs: ' +
                    str(val_loss_difference_percentage_epochs))

        if val_loss_difference_percentage_epochs < self.kwargs[
            'growing_threshold_epochs']:
            return True

        return False


class FireflyWhen(WhenToGrow):
    def when_to_grow(self, history, flops_history, **kwargs) -> bool:
        raise NotImplementedError()


class AlwaysWhen(WhenToGrow):
    def when_to_grow(self, history, flops_history, **kwargs) -> bool:
        """Always returns true

        Args:
            - history (list of tf.keras.callbacks.History)  :
                                    List of previous training history.
            - flops_history (list of int): Flops of the last training  loops
            - kwargs (dict) :   Used to pipe the arguments into the selected method.
                                    This allows methods with different arguments to be
                                    called and allows also futureimplementations/changes.

        Returns:
            (bool)  :   Bool if the when_typel should be grow.
        """
        return True


# Getter for the When methods:


SUPPORTED_WHENS = {'predefined': PredefinedWhen,
                   'autogrow': AutoGrowWhen,
                   'firefly': FireflyWhen,
                   'vallloss_epochs': PredefinedWhenValidationLossEpoch,
                   'always': AlwaysWhen,
                   'vallloss_flops': PredefinedWhenValidationLossFlops}


def get_when_to_grow(*, when_type: str, **kwargs):
    ''' Getter for our supported 'when' methods

    Description:
        - initialises and returns selected when-class
        - throws assertion if when_type is not supported.
        - abstracts init of when from the growingneuralnetworks.py

    '''

    assert when_type.lower() in SUPPORTED_WHENS, \
        f'when_type "{when_type}" is not supported.' \
        f"Supported when_types are:\n {SUPPORTED_WHENS.keys()}"

    return SUPPORTED_WHENS[when_type.lower()](**kwargs)
