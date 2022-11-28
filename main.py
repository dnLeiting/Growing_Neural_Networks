import wandb
import argparse
import yaml
import os
import sys
import logging
from dataclasses import asdict

from nng import Dataloader
from nng.growingneuralnetwork import NNG
from nng.cfg import NNGConfig, DatasetConfig

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=os.environ.get("LOGLEVEL", "INFO"),
)


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    return parser.parse_args(args)


def main():
    args = parse_arguments(sys.argv[1:])

    # Load config file
    logger.info("Loading config...")
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    nng_config = NNGConfig.from_dict(config)
    data_config = DatasetConfig.from_dict(config)

    configs = asdict(nng_config) | asdict(data_config)

    kwargs = {
        arg: config[arg] for arg in config if arg not in configs.keys()
    }

    configs = configs | kwargs

    # Initialize wandb
    if nng_config.wandb:
        logger.info("Initializing  WandB...")
        wandb.init(project=nng_config.project_name,
                   entity="ds-project",
                   config=configs,
                   name=nng_config.run_name)

    # Loading the data
    logger.info("Loading dataset...")
    data_loader = Dataloader(data_config)
    training_ds, validation_ds, test_ds = data_loader.load_data()

    logger.info("Running Growing Neural Network...")

    nng = NNG(
        nng_cfg=nng_config,
        train_ds=training_ds,
        valid_ds=validation_ds,
        test_ds=test_ds,
        **kwargs
    )
    nng.neural_network_growth()


if __name__ == '__main__':
    main()
