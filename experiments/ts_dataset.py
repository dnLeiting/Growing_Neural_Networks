from teacher import TS_Model
import yaml
import tensorflow as tf


def sample_dataset_from_teacher():

    with open('experiments/ts_config.yaml') as file:
        config = yaml.safe_load(file)

    tf.random.set_seed(config['seed'])
    m_i, m_h, m_o = config['teacher']

    teacher = TS_Model(m_i, m_h, m_o, seed=config['seed'])

    data = tf.random.normal(
        shape=(config['num_train_ds']+config['num_test_ds'], m_i), mean=0, stddev=1)
    tf.random.shuffle(data, seed=config['seed'])

    tr_data = data[:config['num_train_ds'], :]
    te_data = data[config['num_train_ds']:, :]

    tr_targets = teacher.predict(tr_data)
    te_targets = teacher.predict(te_data)

    train_ds = tf.data.Dataset.from_tensor_slices((tr_data, tr_targets))
    test_ds = tf.data.Dataset.from_tensor_slices((te_data, te_targets))

    train_ds = train_ds.shuffle(
        config['shuffle_buffer']).batch(config['batch_size'])
    test_ds = test_ds.batch(config['batch_size'])

    return train_ds, test_ds, tr_data, tr_targets
