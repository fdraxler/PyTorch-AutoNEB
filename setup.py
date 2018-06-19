from distutils.core import setup

setup(name='PyTorch-AutoNEB',
      version='1.0',
      description='PyTorch AutoNEB implementation to identify minimum energy paths, e.g. in neural network loss landscapes (https://arxiv.org/abs/1803.00885)',
      author='Felix Draxler',
      author_email='felix.draxler@iwr.uni-heidelberg.de',
      packages=['torch_autoneb'], requires=['networkx', 'torch'])
