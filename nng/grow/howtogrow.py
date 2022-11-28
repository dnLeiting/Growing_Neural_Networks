import logging
from typing import List

import numpy as np
from numpy import ndarray

logger = logging.getLogger(__name__)


# for e.g. GRADMAX. Maybe we can collect these and the random ones in a basic
# How function that takes a keywork such as 'random', 'zeros', 'ones' as input.

class HowToGrow:
    """Implements the how methods.

    Args:
        model (tf.keras.Model)  :   Model to train.
        number_of_new_neurons (int) :   Number of new neurons which are added
                                        to a layer.
        layer (int) :   The layer where the new weights should be added.
        init_seed (int) :   The seed used to initialize the weights generation.

    """

    def __init__(self, seed=42,
                 **kwargs):
        self.init_seed = seed

    def how_to_grow(self, *,
                    bck_shape: tuple[int],
                    **kwargs) -> list[ndarray, ndarray]:
        raise NotImplementedError()


class ConstantHow(HowToGrow):
    def how_to_grow(self, *, bck_shape: tuple[int], const: float, **kwargs):
        """

        @param bck_shape:
        @param const:
        @param kwargs:
        @return:
        """
        del kwargs
        new_w_block = const * np.ones(shape=bck_shape)
        new_b_block = const * np.ones(shape=(bck_shape[1],))
        return [new_w_block, new_b_block]


class GradMaxHow(HowToGrow):
    def how_to_grow(self) -> np.ndarray:
        raise NotImplementedError()


class AutoGrowHowGauInit(HowToGrow):
    def how_to_grow(self, *,
                    bck_shape: tuple[int],
                    old_wghts: list[ndarray],
                    seed=42,
                    **kwargs) -> List[ndarray]:
        ''' Function for sampling gaussian distribution matrices.

        Description:
            Given current layer weights and a shape this function creates a
            new matrix-block of the given shape to be inserted into the
            existing layer during the growing.
            The entries of the new matrix are sampled from a normal
            distribution whose parameters are loc=0.0 and scale=1.0.

        Args.:
            - bck_shape (tuple[int])    :   shape of the new block matrix
            - old_wghts (list[ndarray]) :   first element: weight matrix
                                            second element: bias matrix

        Returns:
            List containing     (first)     the generated weight matrix block,
                                (second)    the generated bias matrix block.
        '''

        new_w_block = np.random.normal(loc=0.0, scale=1.0, size=bck_shape)
        new_b_block = np.random.normal(loc=0.0,
                                       scale=1.0,
                                       size=(bck_shape[-1],))

        return [new_w_block, new_b_block]


class AutoGrowHowZeroInit(HowToGrow):
    def how_to_grow(self, *,
                    bck_shape: tuple[int],
                    old_wghts: list[ndarray],
                    seed=42,
                    **kwargs) -> List[ndarray]:
        ''' Function for sampling autogrow zero matrices.

        Description:
            Given current layer weights and a shape this function creates a
            new matrix-block of the given shape to be inserted into the
            existing layer during the growing.
            The entries of the new matrix are zeros.

        Args.:
            - bck_shape (tuple[int])    :   shape of the new block matrix
            - old_wghts (list[ndarray]) :   first element: weight matrix
                                            second element: bias matrix

        Returns:
            List containing     (first)     the generated weight matrix block,
                                (second)    the generated bias matrix block.
        '''
        new_w_block = np.zeros(shape=bck_shape)
        new_b_block = np.zeros(shape=bck_shape[-1])

        return [new_w_block, new_b_block]


class AutoGrowHowUniInit(HowToGrow):
    def how_to_grow(self, *,
                    bck_shape: tuple[int],
                    old_wghts: list[ndarray],
                    seed=42,
                    **kwargs) -> List[ndarray]:
        ''' Function for sampling a uniform matrices.

        Description:
            Given current layer weights and a shape this function creates a
            new matrix-block of the given shape to be inserted into the
            existing layer during the growing.
            The entries of the new matrix are sampled from a uniform
            distribution whose parameters are low=0.0, and high=1.0.

        Args.:
            - bck_shape (tuple[int])    :   shape of the new block matrix
            - old_wghts (list[ndarray]) :   first element: weight matrix
                                            second element: bias matrix

        Returns:
            List containing     (first)     the generated weight matrix block,
                                (second)    the generated bias matrix block.
        '''
        new_w_block = np.random.uniform(low=-1.0, high=1.0, size=bck_shape)
        new_b_block = np.random.uniform(low=-1.0,
                                        high=1.0,
                                        size=(bck_shape[-1],))

        return [new_w_block, new_b_block]


class RandomHow(HowToGrow):
    def how_to_grow(self, *,
                    bck_shape: tuple[int],
                    old_wghts: list[ndarray],
                    seed=42,
                    **kwargs) -> List[ndarray]:
        ''' Function for sampling random matrices.

        Description:
            Given current layer weights and a shape this function creates a 
            new matrix-block of the given shape to be inserted into the 
            existing layer during the growing.
            The entries of the new matrix are sampled from a normal
            distribution whose parameters are sampled from the existsing 
            weights.

        Args.:
            - bck_shape (tuple[int])    :   shape of the new block matrix
            - old_wghts (list[ndarray]) :   first element: weight matrix
                                            second element: bias matrix

        Returns:
            List containing     (first)     the generated weight matrix block,
                                (second)    the generated bias matrix block.
        '''

        if old_wghts:
            old_w = old_wghts[0]
            mean_w = old_w.mean()
            std_w = old_w.std()

            old_b = old_wghts[1]
            mean_b = old_b.mean()
            std_b = old_b.std()
        else:
            mean_w, mean_b = 0, 0
            std_w, std_b = 1, 1

        np.random.seed(seed)

        logger.info('mean_w: ' + str(mean_w) + ', std_w: ' + str(std_w))
        logger.info('mean_b: ' + str(mean_b) + ', std_b: ' + str(std_b))
        new_w_block = np.random.normal(loc=mean_w, scale=std_w, size=bck_shape)
        new_b_block = np.random.normal(loc=mean_b,
                                       scale=std_b,
                                       size=(bck_shape[-1],))

        return [new_w_block, new_b_block]


SUPPORTED_HOWS = {
    'random_baseline': RandomHow,
    'gradmax': GradMaxHow,
    'const': ConstantHow,
    'autogrow_zeroinit': AutoGrowHowZeroInit,
    'autogrow_uniinit': AutoGrowHowUniInit,
    'autogrow_gauinit': AutoGrowHowGauInit
}


def get_how_to_grow(*, how_type: str, **kwargs):
    assert how_type.lower() in SUPPORTED_HOWS, \
        f'how_type "{how_type}" is not supported.' \
        f"Supported how_types are:\n {SUPPORTED_HOWS.keys()}"

    return SUPPORTED_HOWS[how_type.lower()](**kwargs)
