import keras
from teacher import TS_Model
import yaml
from ts_dataset import sample_dataset_from_teacher
from wandb.keras import WandbCallback
import wandb
import logging

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

arch = 'big_baseline'

# WANDB
if config["wandb"]:
    wandb.init(
        project="TS_baseline",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name="TS_{}_seed_{}".format(arch, config['seed']),
        # Track hyperparameters and run metadata
        entity="ds-project",
        config=config)


train_ds, test_ds, tr_data, tr_targets = sample_dataset_from_teacher()

optimizer = getattr(keras.optimizers, config['optimizer'])(config['lr'])
loss_fn = getattr(keras.losses, 'MeanSquaredError')()

# HERE YOU SWITCH FROM SMALL TO BIG BASELINE!!!
m_i, m_h, m_o = config[arch]

model = TS_Model(m_i, m_h, m_o, special_init=False)  


model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=[],
              )

# -----------------------------------------------------------------------------
#   TRAIN
# -----------------------------------------------------------------------------

hists = list()


for i in range(config['max_epoch']):

    logging.info(f"\n\n>>>>>> {i}/{config['max_epoch']} <<<<<<<<\n\n\n")

    history = model.fit(train_ds,
                        epochs=1,
                        validation_data=test_ds,
                        callbacks=[WandbCallback()] if config['wandb'] else [])
    # raises error something wrong with the dataloader?
    model.evaluate(test_ds)

    hists.append(history)
