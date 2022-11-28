from unittest import TestCase

from keras.layers import Dense
from tensorflow import keras

from nng.grow.howtogrow import get_how_to_grow


class TestRandomHow(TestCase):
    def test_how_to_grow(self):
        """Function that tests the method "how_to_grow" in the RandomHow class

        """
        how = get_how_to_grow(how_type='random_baseline')
        n_in, n_neu = 12, 30
        kwargs = dict()
        w, b = how.how_to_grow(bck_shape=(
            n_in, n_neu), old_wghts=None, **kwargs)

        print(w.shape)
        print(b.shape)

        lay = Dense(23)
        lay(keras.Input((1, 12)))
        old_wghts = lay.get_weights()
        print(type(old_wghts[0]))

        how = get_how_to_grow(how_type='random_baseline')
        n_in, n_neu = 12, 30
        kwargs = dict()
        w, b = how.how_to_grow(bck_shape=(
            n_in, n_neu), old_wghts=old_wghts, **kwargs)
        print(w.shape)
        print(b.shape)


class TestAutoGrowHowZeroInit(TestCase):
    def test_how_to_grow(self):
        """Function that tests the method "how_to_grow" in the AutoGrowHowZeroInit class

        """
        how = get_how_to_grow(how_type='autogrow_zeroinit')
        n_in, n_neu = 12, 30
        kwargs = dict()
        w, b = how.how_to_grow(bck_shape=(
            n_in, n_neu), old_wghts=None, **kwargs)

        print(w.shape)
        print(b.shape)

        lay = Dense(23)
        lay(keras.Input((1, 12)))
        old_wghts = lay.get_weights()
        print(type(old_wghts[0]))

        how = get_how_to_grow(how_type='autogrow_zeroinit')
        n_in, n_neu = 12, 30
        kwargs = dict()
        w, b = how.how_to_grow(bck_shape=(
            n_in, n_neu), old_wghts=old_wghts, **kwargs)
        print(w.shape)
        print(b.shape)


class TestAutoGrowHowUniInit(TestCase):
    def test_how_to_grow(self):
        """Function that tests the method "how_to_grow" in the AutoGrowHowUniInit class

        """
        how = get_how_to_grow(how_type='autogrow_uniinit')
        n_in, n_neu = 12, 30
        kwargs = dict()
        w, b = how.how_to_grow(bck_shape=(
            n_in, n_neu), old_wghts=None, **kwargs)

        print(w.shape)
        print(b.shape)

        lay = Dense(23)
        lay(keras.Input((1, 12)))
        old_wghts = lay.get_weights()
        print(type(old_wghts[0]))

        how = get_how_to_grow(how_type='autogrow_uniinit')
        n_in, n_neu = 12, 30
        kwargs = dict()
        w, b = how.how_to_grow(bck_shape=(
            n_in, n_neu), old_wghts=old_wghts, **kwargs)
        print(w.shape)
        print(b.shape)


class TestAutoGrowHowGauInit(TestCase):
    def test_how_to_grow(self):
        """Function that tests the method "how_to_grow" in the AutoGrowHowGauInit class

        """
        how = get_how_to_grow(how_type='autogrow_gauinit')
        n_in, n_neu = 12, 30
        kwargs = dict()
        w, b = how.how_to_grow(bck_shape=(
            n_in, n_neu), old_wghts=None, **kwargs)

        print(w.shape)
        print(b.shape)

        lay = Dense(23)
        lay(keras.Input((1, 12)))
        old_wghts = lay.get_weights()
        print(type(old_wghts[0]))

        how = get_how_to_grow(how_type='autogrow_gauinit')
        n_in, n_neu = 12, 30
        kwargs = dict()
        w, b = how.how_to_grow(bck_shape=(
            n_in, n_neu), old_wghts=old_wghts, **kwargs)
        print(w.shape)
        print(b.shape)


class TestConstwHow(TestCase):
    """Simple Test suite for the constant updater which loads weights of the 
            predefined constant value.
            The resulting matrices are displayed for manual assessment.
    """

    def test_const_how_to_grow(self):
        how = get_how_to_grow(how_type='const')
        n_in, n_neu = 12, 30
        kwargs = dict()
        w, b = how.how_to_grow(bck_shape=(
            n_in, n_neu), const=3.14159265, **kwargs)

        print('w', w[0, 0], w.shape)
        print('b', b[0], b.shape)
