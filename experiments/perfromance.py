from nng.grow.howtogrow import get_how_to_grow
from nng.grow.whentogrow import get_when_to_grow
from nng.grow.wheretogrow import get_where_to_grow
from nng.grow.models import create_nng
from nng.dataloader import Dataloader
from nng.cfg import DatasetConfig
from datetime import datetime
from nng.train import Trainer
import keras
import wandb


def compile_func(model):
    model.compile(optimizer=getattr(keras.optimizers, 'Adam')(0.001),
                  loss=getattr(keras.losses, 'SparseCategoricalCrossentropy')(from_logits=True),
                  metrics=[getattr(keras.metrics, 'SparseCategoricalAccuracy')()], )


SUPPORTED_WHENS = [
    'predefined',
    'autogrow',
    'firefly',
    'vallloss_epochs',
    'always',
    'vallloss_flops']

SUPPORTED_HOWS = [
    'random_baseline',
    'const',
    'autogrow_zeroinit',
    'autogrow_uniinit',
    'autogrow_gauinit'
]

SUPPORTED_WHERES = [
    'add_5_with_random_layer',
    'add_5',
    'autogrow',
]

data_loader = Dataloader(DatasetConfig())
training_ds, validation_ds, test_ds = data_loader.load_data()

model = create_nng(model_type="MLP", out_dim=10, arch=[5, 5, 5, 5, 5])
compile_func(model)
training_history = []
train = Trainer(model, training_ds, validation_ds, 25, False)
history, model = train.train(model)
training_history.append(history)
lays = model.layers
lay_idx = 1
lay_in = lays[lay_idx]
lay_out = lays[lay_idx + 1]

old_wghts_in = lay_in.get_weights()
old_wghts_out = lay_out.get_weights()

n_in, _ = old_wghts_in[0].shape
_, n_out = old_wghts_out[0].shape

flops_history = [10000]*len(training_history)


kwargs = {
    'model': model,
    "history": training_history,
    "flops_history": flops_history,
    "bck_shape": (n_in, n_in * 2),
    "old_wghts": old_wghts_in,
    "const": 0.5,  # no example in cfg dir
    "max_training": 100,
    "val_sparse_categorical_accuracy": 0.06,  # from default config file
    "growing_threshold_epochs": 0.05,  # from default config file
    "growing_threshold_flops": 0.2,  # from default config file
    "neuron_growth": 10,  # from default config file
    "where_autogrow_policy": "ConvergentGrowth",
    "tau": 0.01,
    "J": 30,
}

wandb.init(project="performance", entity="ds-project")
wandb_data = []
N = 10000

for how in SUPPORTED_HOWS:
    method = get_how_to_grow(how_type=how, **kwargs)
    start = datetime.now()
    for _ in range(N):
        method.how_to_grow(**kwargs)
    end = datetime.now()
    wandb_data.append(["HOW", "HOW_"+method.__class__.__name__, (end-start).microseconds/N, N])

for when in SUPPORTED_WHENS:
    method = get_when_to_grow(when_type=when, **kwargs)
    start = datetime.now()
    for _ in range(N):
        method.when_to_grow(**kwargs)
    end = datetime.now()
    wandb_data.append(["WHEN", "WHEN_"+method.__class__.__name__, (end-start).microseconds/N, N])

for where in SUPPORTED_WHERES:
    method = get_where_to_grow(where_type=where, **kwargs)
    start = datetime.now()
    for _ in range(N):
        method.where_to_grow(**kwargs)
    end = datetime.now()
    wandb_data.append(["WHERE", "WHERE_"+method.__class__.__name__, (end-start).microseconds/N, N])

my_table = wandb.Table(
    columns=["mode", "method", "avg_time[μs]", "N"],
    data=wandb_data)
wandb.log({"Performance": my_table, "N": N})
