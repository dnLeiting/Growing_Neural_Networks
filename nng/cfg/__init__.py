from dataclasses import dataclass, field, fields


@dataclass
class BaseConfig:
    @classmethod
    def from_dict(cls, obj: dict):
        fields_names = [fld.name for fld in fields(cls)]
        dct = {k: v for (k, v) in obj.items() if k in fields_names}
        return cls(**dct)


@dataclass
class NNGConfig(BaseConfig):
    # Log
    wandb: bool = True
    plot: bool = True
    project_name: str = 'test-project'
    run_name: str = 'default_name'
    normalized: bool = False

    # Training
    epochs: int = 5
    max_epochs: int = 20
    seed: int = 42
    optimiser: str = 'Adam'
    init_architecture: list[int] = field(default_factory=lambda: [5])
    activation_function: str = 'relu'
    model_type: str = 'MLP'
    learning_step: float = 0.001
    loss_function: str = 'SparseCategoricalCrossentropy'
    metric: str = 'SparseCategoricalAccuracy'
    neuron_growth: int = 10
    growing: bool = True

    # Methods
    when_type: str = 'predefined'
    where_type: str = 'add_predefined'
    how_type: str = 'random_baseline'
    grow_neu_type: str = 'adding'
    grow_lay_type: str = 'baseline'

@dataclass
class DatasetConfig(BaseConfig):
    dataset: str = 'mnist'
    num_classes: int = 10
    batch_size: int = 128
    split_rate: list[int] = field(default_factory=lambda: [80, 19, 1])
    seed: int = 42
