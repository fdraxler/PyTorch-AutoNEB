import unittest
from os import listdir
from unittest import TestCase, skipUnless

import torch
from networkx import MultiGraph
from os.path import join, dirname
from torch import cuda
from torch.nn import NLLLoss
from torch.optim import Adam, SGD
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Pad, Compose
from yaml import safe_load

from main import setup_project, read_config_file
from torch_autoneb import find_minimum, neb, suggest_pair, auto_neb
from torch_autoneb.config import OptimConfig, EvalConfig, NEBConfig, AutoNEBConfig, LandscapeExplorationConfig
from torch_autoneb.datasets import XORDataset
from torch_autoneb.fill import equal, highest
from torch_autoneb.models import CompareModel, DataModel, ModelWrapper, CNN, DenseNet, ResNet, Eggcarton
from torch_autoneb.models.mlp import MLP
from torch_autoneb.models.network_input import NetworkInputModel
from torch_autoneb.suggest import disconnected, unfinished, mst


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
        neb_config = NEBConfig(float("inf"), 1e-5, equal, {"count": 3}, 1, neb_optim_config)

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
        weight_decay = 0
        subsample_pivot_count = 1
        neb_configs = [
            NEBConfig(spring_constant, weight_decay, equal, {"count": 2}, subsample_pivot_count, optim_config_1),
            NEBConfig(spring_constant, weight_decay, highest, {"count": 3, "key": "dense_train_loss"}, subsample_pivot_count, optim_config_1),
            NEBConfig(spring_constant, weight_decay, highest, {"count": 3, "key": "dense_train_loss"}, subsample_pivot_count, optim_config_2),
            NEBConfig(spring_constant, weight_decay, highest, {"count": 3, "key": "dense_train_loss"}, subsample_pivot_count, optim_config_2),
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


class TestModels(TestCase):
    @classmethod
    def setUpClass(cls):
        transform = Compose([Pad(2), ToTensor()])
        cls.train_mnist = MNIST(join(dirname(__file__), "tmp/mnist"), True, transform, download=True)
        cls.test_mnist = MNIST(join(dirname(__file__), "tmp/mnist"), False, transform, download=True)
        cls.input_size = cls.train_mnist[0][0].shape
        cls.output_size = 10
        cls.random_error = 1 - (1 / cls.output_size)

    def _test_model(self, nn_model):
        loss_model = CompareModel(nn_model, NLLLoss())
        data_model = DataModel(loss_model, {
            "train": self.train_mnist,
            "test": self.test_mnist
        })
        model = ModelWrapper(data_model)
        model.adapt_to_config(EvalConfig(1024))

        if cuda.is_available():
            pass  # model.to("cuda")
        analysis = model.analyse()

        for offset, data, size, is_buffer in model.iterate_params_buffers():
            print(size)

        for key, value in analysis.items():
            if "error" in key:
                self.assertLess(self.random_error * 0.9, value, f"Random {key} too low")
                self.assertGreater(self.random_error * 1.1, value, f"Random {key} too high")
        print(nn_model.__class__.__name__, analysis)

    def test_mlp(self):
        mlp = MLP(2, 10, self.input_size, self.output_size)
        self._test_model(mlp)

    def test_cnn(self):
        cnn = CNN(2, 12, 3, 1, 2, 1, 1, 16, self.input_size, self.output_size)
        self._test_model(cnn)

    def test_densenet(self):
        densenet = DenseNet(12, 100, 1 / 2, True, self.input_size, self.output_size)
        self._test_model(densenet)

    def test_resnet(self):
        resnet = ResNet(20, self.input_size, self.output_size)
        self._test_model(resnet)


class TestMain(TestCase):
    def test_parse_configs(self):
        base_dir = join(dirname(__file__), "configs")
        for config_file in listdir(base_dir):
            if config_file.endswith(".yaml"):
                with open(join(base_dir, config_file), "r") as file:
                    config_dict = safe_load(file)
                    config = LandscapeExplorationConfig.from_dict(config_dict["exploration"])
                    self.assertIsInstance(config, LandscapeExplorationConfig)

    def test_project_management(self):
        project_dirname = join(dirname(__file__), "tmp", "project")
        config_file = join(dirname(__file__), "configs", "test.yaml")
        setup_project(config_file, project_dirname)

    def test_read_config(self):
        config_file = join(dirname(__file__), "configs", "test.yaml")
        model, minima_count, min_config, lex_config = read_config_file(config_file)
        self.assertIsInstance(model, ModelWrapper)
        self.assertGreater(minima_count, 0)
        self.assertIsInstance(min_config, OptimConfig)
        self.assertIsInstance(lex_config, LandscapeExplorationConfig)


class LimitTest(TestCase):
    def test_long_run(self):
        eggcarton = Eggcarton(2)
        model = ModelWrapper(eggcarton)
        minima = [find_minimum(model, OptimConfig(1000, SGD, {"lr": 0.1}, None, None, None)) for _ in range(2)]

        neb_optim_config = OptimConfig(1000, SGD, {"lr": 0.1}, None, None, None)
        neb_config = NEBConfig(float("inf"), 1e-5, equal, {"count": 20}, 1, neb_optim_config)
        neb({
            "path_coords": torch.cat([m["coords"].view(1, -1) for m in minima]),
            "target_distances": torch.ones(1)
        }, model, neb_config)


class InputModelTest(TestCase):
    def test_dataset_generation(self):
        transform = Compose([Pad(2), ToTensor()])
        train_mnist = MNIST(join(dirname(__file__), "tmp/mnist"), True, transform, download=True)
        input_size = train_mnist[0][0].shape
        number_of_classes = 10
        resnet = ResNet(20, input_size, number_of_classes)

        # Find a minimiser for the network
        optim_wrapper = ModelWrapper(DataModel(CompareModel(resnet, NLLLoss()), {"train": train_mnist}))
        optim_wrapper.to("cuda")
        optim_config = OptimConfig(100, SGD, {"lr": 0.1}, None, None, EvalConfig(128))
        minimum = find_minimum(optim_wrapper, optim_config)
        optim_wrapper.set_coords_no_grad(minimum["coords"])

        nim = NetworkInputModel(resnet, input_size, 0)
        nim.cuda()
        resnet.cuda()
        dataset = nim.generate_dataset(train_mnist, number_of_classes)
        self.assertEqual(len(dataset), 100)


if __name__ == '__main__':
    unittest.main()
