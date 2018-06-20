import unittest
from itertools import product, repeat
from unittest import TestCase, skipUnless

import torch
from networkx import MultiGraph
from torch import FloatTensor, LongTensor, normal
from torch.nn import NLLLoss
from torch.optim import Adam, SGD
from torch.utils.data import Dataset

from torch_autoneb import OptimConfig, find_minimum, neb, suggest_pair, auto_neb
from torch_autoneb.fill import equal, highest
from torch_autoneb.hyperparameters import EvalConfig, NEBConfig, AutoNEBConfig, LandscapeExplorationConfig
from torch_autoneb.models import CompareModel, DataModel, ModelWrapper
from torch_autoneb.models.mlp import MLP
from torch_autoneb.suggest import disconnected, unfinished, mst


class XORDataset(Dataset):
    def __init__(self, train, transform=None, target_transform=None):
        self.train = train

        if train:
            size = 500
        else:
            size = 100
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


def _create_xor_model():
    mlp = MLP(2, 10, 2, 2, False)
    loss_model = CompareModel(mlp, NLLLoss())
    data_model = DataModel(loss_model, {
        "train": XORDataset(train=True),
        "test": XORDataset(train=False)
    })
    model = ModelWrapper(data_model)
    return model


class TestAlgorithms(TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        super(TestAlgorithms, cls).setUpClass()
        cls.model = _create_xor_model()
        cls.model.to(cls.device)

        min_eval_config = EvalConfig(128)
        min_optim_config = OptimConfig(100, Adam, {}, None, None, min_eval_config)
        cls.minima = [find_minimum(cls.model, min_optim_config) for _ in range(2)]

    def test_find_minimum(self):
        result = self.minima[0]

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
        minima = self.minima[:2]

        neb_eval_config = EvalConfig(128)
        neb_optim_config = OptimConfig(10, Adam, {}, None, None, neb_eval_config)
        neb_config = NEBConfig(float("inf"), equal, {"count": 3}, 1, neb_optim_config)

        result = neb({
            "path_coords": torch.cat([m["coords"].view(1, -1) for m in minima]),
            "target_distances": torch.ones(1)
        }, self.model, neb_config)

        required_keys = [
            "path_coords",
            "target_distances",
            "saddle_train_error",
            "saddle_train_loss",
            "saddle_test_error",
            "saddle_test_loss",
            "dense_train_error",
            "dense_train_loss",
            "dense_test_error",
            "dense_test_loss",
        ]
        for key in required_keys:
            self.assertTrue(key in result, f"{key} not in result")
            value = result[key]
            self.assertFalse(torch.isnan(value).any().item(), f"{key} contains a NaN value")
            if "saddle_" in key:
                print(key, value.item())

    def test_auto_neb(self):
        # Test AutoNEB procedure
        graph = MultiGraph()
        for idx, minimum in enumerate(self.minima):
            graph.add_node(idx + 1, **minimum)

        # Set up AutoNEB schedule
        spring_constant = float("inf")
        eval_config = EvalConfig(128)
        optim_config_1 = OptimConfig(10, SGD, {"lr": 0.1}, None, None, eval_config)
        optim_config_2 = OptimConfig(10, SGD, {"lr": 0.01}, None, None, eval_config)
        neb_configs = [
            NEBConfig(spring_constant, equal, {"count": 2}, 1, optim_config_1),
            NEBConfig(spring_constant, highest, {"count": 3, "key": "dense_train_loss"}, 1, optim_config_1),
            NEBConfig(spring_constant, highest, {"count": 3, "key": "dense_train_loss"}, 1, optim_config_2),
            NEBConfig(spring_constant, highest, {"count": 3, "key": "dense_train_loss"}, 1, optim_config_2),
        ]
        auto_neb_config = AutoNEBConfig(neb_configs)
        self.assertEqual(auto_neb_config.cycle_count, len(neb_configs))

        # Run AutoNEB
        auto_neb(1, 2, graph, self.model, auto_neb_config)
        self.assertEqual(len(graph.edges), auto_neb_config.cycle_count)


@skipUnless(torch.cuda.is_available(), "Cuda is not available")
class TestCudaAlgorithms(TestAlgorithms):
    device = "cuda"


class TestSuggestEngines(TestCase):
    def test_sequence(self):
        graph = MultiGraph()
        graph.add_node(1, value=1)  # Global minimum
        graph.add_node(2, value=2)
        graph.add_node(3, value=3)
        graph.add_node(4, value=4)

        def weight(id_pair):
            return sum(node ** 2 for node in id_pair)

        unfinished_edge = (1, 3)
        config = LandscapeExplorationConfig("value", "weight", [], None, AutoNEBConfig([None, None]))

        # Disconnect suggest
        correct_order = [
            (1, 2), (1, 3), (1, 4),
        ]
        config.suggest_methods = [disconnected]
        while True:
            pair = suggest_pair(graph, config)
            if pair[0] is None:
                break
            self.assertGreater(len(correct_order), 0, "disconnected_suggest gives more pairs than necessary")
            assert pair == correct_order.pop(0)
            graph.add_edge(*pair, key=1, weight=weight(pair))
            if pair != unfinished_edge:
                # Skip the second edge for this pair, to test unfinished suggest
                graph.add_edge(*pair, key=2, weight=weight(pair))
        self.assertEqual(len(correct_order), 0, "disconnected_suggest missing suggestions!")

        # Unfinished suggest
        correct_order = [
            unfinished_edge
        ]
        config.suggest_methods = [unfinished]
        while True:
            pair = suggest_pair(graph, config)
            if pair[0] is None:
                break
            self.assertGreater(len(correct_order), 0, "unfinished_suggest gives more pairs than necessary")
            assert pair == correct_order.pop(0)
            graph.add_edge(*pair, key=2, weight=weight(pair))
        self.assertEqual(len(correct_order), 0, "unfinished_suggest missing suggestions!")

        # Core: MST suggest
        correct_order = [
            (2, 4), (3, 4),  # Replace (1, 4)
            (2, 3),  # Replace (1, 3)
        ]
        config.suggest_methods = [mst]
        while True:
            pair = suggest_pair(graph, config)
            if pair[0] is None:
                break
            self.assertGreater(len(correct_order), 0, "mst_suggest gives more pairs than necessary")
            assert pair == correct_order.pop(0)
            graph.add_edge(*pair, key=1, weight=weight(pair))
            graph.add_edge(*pair, key=2, weight=weight(pair))
        self.assertEqual(len(correct_order), 0, "mst_suggest missing suggestions!")


if __name__ == '__main__':
    unittest.main()
