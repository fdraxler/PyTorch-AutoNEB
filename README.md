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

Setup your environment, e.g. using

```
conda install pytorch torchvision -c pytorch
```

Optional, but recommended: Install `tqdm` top geht progress bars while running:

```
conda install tqdm
```

Download/Clone the code using

```
git clone https://github.com/fdraxler/PyTorch-AutoNEB
cd PyTorch-AutoNEB
```

## Usage

### Running the examples

```
python main.py project_directory config_file
```

where `project_directory` is the directory (need not exist) where the data should be stored.
`config_file` should point to one of the `.yaml` files in [configs](configs).

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


## Results

The final MSTs for analysis with [Evaluate.ipynb] can be found at [https://drive.google.com/drive/folders/1VQvecH3lWntBD5H0VHyYU577DunvkQlb](this repository).
As of now, it contains only a subset of systems. [https://github.com/fdraxler/PyTorch-AutoNEB/issues/new](Open an issue) to request more systems.
