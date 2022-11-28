import logging

import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import wandb
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

logger = logging.getLogger(__name__)


def log_analysis(dataset: tf.data.Dataset, model: keras.Model, commit: bool = True):
    """
    Method to calculate confusion matrix for multi class classification\
    and metrics such as precision, recall and F1.
    Wandb needed.

    Args:
        dataset: (tf.data.Dataset) dataset to analyse results.
        model: (keras.Model) for calculating number of flops.
        commit: (bool) if it's last log in epoch than commit True if not false.

    Returns:
        int: number of flops.
    """
    predict_y = model.predict(dataset)
    y_pred = np.argmax(predict_y, axis=1)
    labels = list(map(lambda x: x[1], dataset))
    y_true = np.array(tf.concat(labels, axis=0))
    cf_matrix = confusion_matrix(y_true, y_pred)
    sns.set(font_scale=1.3)
    plt.subplots(figsize=(15, 5))
    sns.heatmap(cf_matrix, annot=True)
    wandb.log({"confusion_matrix": wandb.Image(plt)}, step=wandb.run.step, commit=False)
    plt.close()
    plt.subplots(figsize=(15, 5))
    sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
    wandb.log({"confusion_matrix_%": wandb.Image(plt)}, step=wandb.run.step, commit=False)
    plt.close()
    report = classification_report(y_true, y_pred, digits=3, output_dict=True)
    wandb.log(report, step=wandb.run.step, commit=commit)


def get_flops(model: keras.Model, datashape: list):
    """
    Method that calculates the number of FLOPS in the given model for one batch.
    Args:
        model: (keras.Model) for calculating number of flops.
        datashape: (list) of data shape for the model.

    Returns:
        int: number of flops.
    """
    forward_pass = tf.function(
        model.call,
        input_signature=[tf.TensorSpec(shape=(1,) + keras.Input(datashape).shape[1:])])

    graph_info = profile(forward_pass.get_concrete_function().graph,
                         options=ProfileOptionBuilder.float_operation())

    # We can divide it by 2. We can do it, because `profile` counts multiply and accumulate
    # as two flops. If you want to report the total number of multiply accumulate ops than add //2 to return
    return graph_info.total_float_ops
