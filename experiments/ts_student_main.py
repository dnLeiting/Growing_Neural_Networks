from nng.grow.modeltransformer import ModelTransformer
import keras
import yaml
from ts_dataset import sample_dataset_from_teacher
from wandb.keras import WandbCallback
import wandb
import logging
from nng.grow.models import create_nng_student_mlp

logging.getLogger().setLevel(logging.INFO)


# -----------------------------------------------------------------------------
#   FILE TO FOR THE STUDENT-TEACHER EXPERIMENTS
# -----------------------------------------------------------------------------
#
#   - We adapt the approach presented in the GradMax paper.
#   - We created a Teacher network constisting of one hidden layer with
#       m_i = 20 : inputs
#       m_h = 10 : hidden neurons
#       m_o = 10 : outputs
#       see the teacher.py file.
#   - This model is initialized with uniform distributed weights in [-1, 1]
#   - We then sampled 1_000 data points from IR^{m_i} using a normal
#       distribution with zero mean and unit std. From this the training data
#       is generated that is used here.
#   - We implemented two baselines:
#       1. small_baseline    m_i, m_h, m_o : 20, 5, 10
#       2. big_baseline      m_i, m_h, m_o : 20, 10, 10
#       these are trained without growth to obtain an upper and lower bound.
#   - Here we train and track student models that are grown.


# -----------------------------------------------------------------------------
#   CONFIGURATION
# -----------------------------------------------------------------------------

#   which how method to choose
#   how many steps to train
#   setting up wandb

with open('experiments/ts_config.yaml') as f:
    config = yaml.safe_load(f)

if config['wandb']:
    wandb.init(
        project="TS_baseline",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{config['grow_neu_type']}_{config['how_type']}_{config['seed']}",
        # Track hyperparameters and run metadata
        entity="ds-project",
        config=config)


train_ds, test_ds, tr_data, tr_targets = sample_dataset_from_teacher()

optimizer = getattr(keras.optimizers, config['optimizer'])(config['lr'])
loss_fn = getattr(keras.losses, 'MeanSquaredError')()


transformer = ModelTransformer(config['optimizer'],
                               config['lr'],
                               config['loss_fn'],
                               config['metric'])


# model
m_i, m_h, m_o = config['seed_arch']

kernel_init = None
bias_init = None

model = create_nng_student_mlp(
    m_i=m_i, m_h=m_h, m_o=m_o, kernel_init=None, bias_init=None)


def compile(a):
    a.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=[],
              )


compile(model)
model(keras.Input((m_i,)))

# -----------------------------------------------------------------------------
#   TRAIN
# -----------------------------------------------------------------------------
# Build model for wandb callback
build_in = [m_i] + m_h[:-1]
for lay, val in zip(model.lays, build_in):
    lay.build(val)
model.last_lay.build(m_h[-1])

hists = list()
vals = list()

for i in range(config['max_epoch']):
    logging.info('\n\nTraining\n')
    # training
    history = model.fit(train_ds,
                        epochs=1,
                        validation_data=test_ds,
                        callbacks=[WandbCallback()] if config['wandb'] else [])
    hists.append(history)
    val = model.evaluate(test_ds)
    vals.append(val)

    logging.info(f"\n{i}/{config['max_epoch']} :\n ", model)

    if i not in config['schedule']:  # replaces when method
        # nothing to grow
        continue

    # growing
    lay_idx = 0
    n_neu = 1

    logging.info('\n\nGrowing\n')

    grower_kwargs = {
        'grow_neu_type': config["grow_neu_type"],
        'model': model,
        'tr_data': tr_data,
        'tr_targets': tr_targets,
        'n_neu': n_neu,
        'lay_idx': lay_idx,
        'how_type': config["how_type"],
        'training_ds': train_ds,
        'loss_function': config["hw_loss_fn"],
        'history': history,
        'from_logits': False
    }
    kwargs = config | grower_kwargs

    transformer.grow_neuron(
        **kwargs)

    compile(model)

print(hists)
print('val', vals)
