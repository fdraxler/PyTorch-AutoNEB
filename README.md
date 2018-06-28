# PyTorch-AutoNEB

This framework implements NEB (Henkelman and Jónsson, 2000)
and AutoNEB (Kolsbjerg, Groves and Hammer, 2016) in PyTorch.
It efficiently finds low energy paths between minima of
arbitrary loss/energy functions.

This framework was developed to be applied to neural networks,
but is truely generic to any (Auto)NEB+Python application.
Several examples for neural network architectures are given.


## Implemented models/loss functions

The following neural network architecture are included:

- simple CNNs and MLPs,
- ResNets,
- DenseNets

They can be applied on MNIST, CIFAR10 and CIFAR100.


## Installation

### Running the examples

Download/Clone the code using

```
git clone https://github.com/fdraxler/PyTorch-AutoNEB
cd PyTorch-AutoNEB
```

and run

```
python main.py project_directory config_file
```

where `project_directory` is the directory (need not exist) where the data should be stored.
`config_file` should point to a .yaml file like the ones in [configs](configs).

You can create new config files by editing an existing, such as `configs/cifar10-resnet20.yaml`.


### Use in your own code

Install the `torch_autoneb` package by running

```
python setup.py
```

in the root directory of this repository. You can then use it in Python via

```python
import torch_autoneb
```
