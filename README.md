# Growing Neural Network

## Description
Choosing the right architecture of you deep neural network for a given problem is not straight forward. With this package we provide a Machine-Learning pipeline that allows you to *grow* your model architecture dynamically during the training. State off the art growing techniques presented e.g. in *GradMax : Growing Neural Networks using Gradient Information* [[1](https://arxiv.org/abs/2201.05125)] are provided and can be used in a *plug-n-play* fashion.

The general pipeline-structure is shown in the [flow chart](#FlowChart) below. 

## Setup Growing Neural Network

For the repository check [GitLab Repository](https://git.tu-berlin.de/david_leiting/ds-project).

### Requirements
- tensorflow>=2.9.1,<3.0.0
- tensorflow_datasets>=4.5.2,<5.0.0
- wandb>=0.12.16,<1.0.0
- scipy>=1.8.1,<2.0.0
- numpy>=1.22.3,<2.0.0
- matplotlib>=3.5.2,<4.0.0


#### Setup
To install nng package in your environment run:
```
pip install -e .
```

### Usage
To use the package run:
```
python3 main.py --config-path <CONFIG_FILE_PATH>
``` 

### Usage in PyCharm
To run all tests and the code in PyCharm the following things need to be considered

1. Set config file parameter \
Edit Configurations -> Parameters -> Add "--config-file <CONFIG_FILE_PATH>"
   
2. Set environment variable https://wandb.ai/  \
Get API Key: Settings -> API keys \
Set API Key: Edit Configurations -> Environment variables -> Add "WANDB_API_KEY=$YOUR_API_KEY"

### Experiments

For running the experiments please checkout the following ReadMe file: \
nng/cfg/final/ReadMe.md

## Installation
In order to install pip package go to the code directory and run:
```
pip install -e .
```
All needed packages will be installed. The Python 3.8 is needed.

## Usage
After installation you can copy [config.yaml](https://git.tu-berlin.de/david_leiting/ds-project/-/tree/main/code/gnn/cfg/config.yaml) file from [code/gnn/cfg](https://git.tu-berlin.de/david_leiting/ds-project/-/tree/main/code/gnn/cfg) directory and paste it in accessible location.
After setting config file for the experiment you can run:
```
growing_neural_network --config-file <PATH_TO_THE_CONFIG_FILE>
```

## Acknowledgement

We borrowed and modified code from [GradMax]. Therefore we would like to thank the authors of these repositeries.


[GradMax]: https://github.com/google-research/growneuron

