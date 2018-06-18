import unittest
from itertools import product, repeat
from unittest import TestCase

import torch
from networkx import MultiGraph
from torch import FloatTensor, LongTensor, normal
from torch.nn import NLLLoss
from torch.utils.data import Dataset

from torch_autoneb import OptimHyperparameters, find_minimum, neb, suggest_pair
from torch_autoneb.fill.equal import FillEqual
from torch_autoneb.hyperparameters import EvalHyperparameters, NEBHyperparameters
from torch_autoneb.models import CompareModel, DataModel, ModelWrapper
from torch_autoneb.models.mlp import MLP
from torch_autoneb.suggest.disconnected import disconnected_suggest
from torch_autoneb.suggest.mst import mst_suggest
from torch_autoneb.suggest.unfinished import create_unfinished_suggest


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
    def test_find_minimum(self):
        model = _create_xor_model()

        eval_config = EvalHyperparameters(128)
        config = OptimHyperparameters(100, "Adam", {}, eval_config)

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
        model = _create_xor_model()

        min_eval_config = EvalHyperparameters(128)
        min_optim_config = OptimHyperparameters(100, "Adam", {}, min_eval_config)
        minima = [find_minimum(model, min_optim_config) for _ in range(2)]

        neb_eval_config = EvalHyperparameters(128)
        neb_optim_config = OptimHyperparameters(100, "Adam", {}, neb_eval_config)
        neb_config = NEBHyperparameters(float("inf"), FillEqual(), neb_optim_config, 3)

        result = neb({
            "path_coords": torch.cat([m["coords"].view(1, -1) for m in minima]),
            "target_distances": torch.ones(1)
        }, model, neb_config)

        required_keys = [
            "path_coords",
            "target_distances",
            "sad_train_error",
            "sad_train_loss",
            "sad_test_error",
            "sad_test_loss",
            "sub_train_error",
            "sub_train_loss",
            "sub_test_error",
            "sub_test_loss",
        ]
        for key in required_keys:
            self.assertTrue(key in result, f"{key} not in result")

    def test_auto_neb(self):
        pass

    def test_suggest_engines(self):
        graph = MultiGraph()
        graph.add_node(1, value=1)  # Global minimum
        graph.add_node(2, value=2)
        graph.add_node(3, value=3)
        graph.add_node(4, value=4)

        def weight(id_pair):
            return sum(node ** 2 for node in id_pair)

        engines = [mst_suggest]
        unfinished_edge = (1, 3)

        # Disconnect suggest
        correct_order = [
            (1, 2), (1, 3), (1, 4),
        ]
        while True:
            pair = suggest_pair(graph, "value", "weight", disconnected_suggest)
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
        unfinished_suggest = create_unfinished_suggest(2)
        while True:
            pair = suggest_pair(graph, "value", "weight", unfinished_suggest)
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
        while True:
            pair = suggest_pair(graph, "value", "weight", mst_suggest)
            if pair[0] is None:
                break
            self.assertGreater(len(correct_order), 0, "mst_suggest gives more pairs than necessary")
            assert pair == correct_order.pop(0)
            graph.add_edge(*pair, key=1, weight=weight(pair))
            graph.add_edge(*pair, key=2, weight=weight(pair))
        self.assertEqual(len(correct_order), 0, "mst_suggest missing suggestions!")


if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestAlgorithms)
    # unittest.TextTestRunner(verbosity=2).run(suite)
