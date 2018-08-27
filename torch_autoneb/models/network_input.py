import foolbox
import numpy as np
from torch import FloatTensor, LongTensor, from_numpy
from torch.autograd import Variable
from torch.nn import Parameter, Module
from torch.nn.functional import softmax, nll_loss, log_softmax
from functools import reduce
from torch_autoneb.helpers import pbar
from operator import mul


class NetworkInputModel(Module):
    LOGIT = "logit"
    LOGIT_CORRECT = "logit_correct"
    LOGIT_CORRECT_INVERSE = "logit_correct_inverse"
    SOFTMAX = "softmax"
    SOFTMAX_CORRECT = "softmax_correct"
    SOFTMAX_CORRECT_INVERSE = "softmax_correct_inverse"
    CROSS_ENTROPY = "cross_entropy"
    ARGMAX = "argmax"

    def __init__(self, base_model: Module, input_size: tuple, label: int, clamp_data: bool = True, stop_in_target_region: bool = False):
        super().__init__()
        # So that it's not registered as a PyTorch Module with parameters
        self.__dict__["_base_model"] = base_model
        self.input_holder = Parameter(FloatTensor(1, *map(int, input_size)))
        self.label = label
        self.clamp_data = clamp_data
        self.stop_in_target_region = stop_in_target_region

    @property
    def base_model(self) -> Module:
        return self.__dict__["_base_model"]

    def initialise_randomly(self):
        raise NotImplementedError("You need to manually prepare a set of minima, have a look at torch_autoneb.models.network_input.NetworkInputModel.generate_dataset()")

    def generate_dataset(self, dataset, number_of_classes, valid_set_size=50, adversarial_set_size=50):
        final_set = []
        adversarial_set = []
        base_model = self.base_model

        # Setup for adversarial samples
        base_model.eval()
        fool_model = foolbox.models.PyTorchModel(base_model, (0, 1), number_of_classes, cuda=self.input_holder.device.type == "cuda")
        criterion = foolbox.criteria.TargetClassProbability(self.label, p=0.99)
        attack = foolbox.attacks.LBFGSAttack(fool_model, criterion)

        for i in pbar(range(valid_set_size + adversarial_set_size), desc="Generating dataset"):
            data, target = dataset[i]
            if target == self.label:
                if len(final_set) < valid_set_size:
                    # Analyse the current image
                    self.input_holder.data[:] = data.view(self.input_holder.shape)
                    user_data = self.analyse()
                    final_set.append((data, user_data))
            else:
                if len(adversarial_set) < adversarial_set_size:
                    # Generate an adversarial
                    adversarial = attack(data.numpy(), label=target)
                    # Skip failed attempts
                    if np.count_nonzero(adversarial == data.numpy()) == reduce(mul, data.shape):
                        print("Failed to generate an adversarial example")
                        continue

                    data = from_numpy(adversarial).view(self.input_holder.shape)
                    self.input_holder.data[:] = data
                    user_data = self.analyse()
                    user_data.update({
                        "adversarial": True,
                        "original_class": target
                    })
                    adversarial_set.append((data, user_data))

        return adversarial_set + final_set

    def forward(self, **kwargs):
        model = self.base_model
        model.eval()

        assert self.input_holder.shape[0] == 1

        if self.clamp_data and self.training:
            self.input_holder.data.clamp_(0, 1)

        soft_pred = model(self.input_holder)
        hard_pred = soft_pred.data.sort(1, True)[1]
        fake_loss = -soft_pred[:, self.label]

        if self.training and self.stop_in_target_region:
            # Manually set gradient to zero
            is_correct_label = hard_pred[0, 0] == self.label
            if is_correct_label:
                probs = softmax(soft_pred, 1).data.cpu()
                # Correct class has 5% margin to second best class
                happy_with_input = probs[0, self.label] - probs[0, hard_pred[0, 1]] > 0.05
            else:
                happy_with_input = False
            if happy_with_input:
                fake_loss = soft_pred.new([fake_loss])

        return fake_loss

    def analyse(self):
        model = self.base_model
        model.eval()
        soft_pred = model(self.input_holder)
        hard_pred = soft_pred.data.sort(1, True)[1]

        analysis = {
            "logit": soft_pred.data.cpu(),
            "logit_correct": soft_pred.data[:, self.label].cpu()[0],
            "logit_correct_inverse": -soft_pred.data[:, self.label].cpu()[0],
            "softmax": softmax(soft_pred, 1).data.cpu(),
            "softmax_correct": softmax(soft_pred, 1).data[:, self.label].cpu()[0],
            "softmax_correct_inverse": 1 - softmax(soft_pred, 1).data[:, self.label].cpu()[0],
            "cross_entropy": nll_loss(log_softmax(soft_pred, 1).cpu(), Variable(LongTensor([self.label]))).item(),
            "argmax": hard_pred.cpu()[0, 0],
        }
        return analysis
