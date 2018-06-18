import unittest
from unittest import TestCase

from torch import FloatTensor, LongTensor, normal
from itertools import product, repeat

from torch.nn import NLLLoss
from torch.utils.data import Dataset

from torch_autoneb import OptimHyperparameters, find_minimum
from torch_autoneb.hyperparameters import EvalHyperparameters
from torch_autoneb.models import CompareModel, DataModel, ModelWrapper
from torch_autoneb.models.mlp import MLP


class XORDataset(Dataset):
    def __init__(self, train, transform=None, target_transform=None):
        self.train = train

        if train:
            size = 50000
        else:
            size = 10000
        each_size = size // 4
        assert each_size * 4 == size

        self.data = FloatTensor(size, 2)
        self.target = LongTensor(size)
        offset = 0
        for x, y in product((-1, 1), (-1, 1)):
            self.data[offset:offset + each_size] = normal(FloatTensor(list(repeat((x, y), each_size))), 0.2)
            self.target[offset:offset + each_size] = 1 if x == y else 0
            offset += each_size

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        target = self.target[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class TestAlgorithms(TestCase):
    def test_find_minimum(self):
        mlp = MLP(2, 10, 2, 2, False)
        loss_model = CompareModel(mlp, NLLLoss())
        data_model = DataModel(loss_model, {
            "train": XORDataset(train=True),
            "test": XORDataset(train=False)
        })
        model = ModelWrapper(data_model)

        eval_config = EvalHyperparameters(128)
        config = OptimHyperparameters(1000, "Adam", {}, eval_config)

        result = find_minimum(model, config)

        required_keys = [
            "coords",
            "train_error",
            "train_loss",
            "test_error",
            "test_loss",
        ]
        for key in required_keys:
            self.assertTrue(key in result, f"{key} not in result")

    def test_neb(self):
        pass

    def test_auto_neb(self):
        pass


if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestAlgorithms)
    # unittest.TextTestRunner(verbosity=2).run(suite)
