import torch.nn as nn
import torch

from networks import resnet, wrn


class TLResNet(resnet.ResNet):
    """
    overwrite function "__init__" and "forward"
    """

    def __init__(self, block, num_block, num_classes=10):
        super(TLResNet, self).__init__(block, num_block, num_classes)

        # freeze convolution layer
        for p in self.parameters():
            p.requires_grad = False
        # unfreeze fc layer
        for p in self.fc.parameters():
            p.requires_grad = True

        # print trainable parameters
        print("trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"name: {name}: size: {param.size()}")

    def load_model(self, model_path):
        model = torch.load(model_path)
        # todo
        # the initialize of fc layer maybe enhance
        model["fc.weight"] = torch.rand_like(self.fc.weight)
        model["fc.bias"] = torch.rand_like(self.fc.bias)

        self.load_state_dict(model)

    @staticmethod
    def create_model(model_name, num_classes=10):
        model_parameters = {
            "resnet18": {
                "block": resnet.BasicBlock,
                "num_block": [2, 2, 2, 2],
            },
            "resnet34": {
                "block": resnet.BasicBlock,
                "num_block": [3, 4, 6, 3]
            },
            "resnet50": {
                "block": resnet.BottleNeck,
                "num_block": [3, 4, 6, 3]
            },
            "resnet101": {
                "block": resnet.BottleNeck,
                "num_block": [3, 4, 23, 3]
            },
            "resnet152": {
                "block": resnet.BottleNeck,
                "num_block": [3, 8, 36, 3]
            },
        }

        model_param = model_parameters.get(model_name)
        if not model_name:
            raise ValueError("incorrect model name")

        model_param.update({"num_classes": num_classes})

        return getattr(TLResNet, model_name)(**model_param)

    @staticmethod
    def resnet18(block, num_block, num_classes):
        return TLResNet(block, num_block, num_classes)

    @staticmethod
    def resnet34(block, num_block, num_classes):
        return TLResNet(block, num_block, num_classes)

    @staticmethod
    def resnet50(block, num_block, num_classes):
        return TLResNet(block, num_block, num_classes)

    @staticmethod
    def resnet101(block, num_block, num_classes):
        return TLResNet(block, num_block, num_classes)

    @staticmethod
    def resnet152(block, num_block, num_classes):
        return TLResNet(block, num_block, num_classes)


class TLWideResNet(wrn.WideResNet):

    def __init__(self, depth, num_classes, widen_factor=1, droprate=0):
        super(TLWideResNet, self).__init__(depth, num_classes, widen_factor, droprate)

        # freeze convolution layer
        for p in self.parameters():
            p.requires_grad = False
        # unfreeze fc layer
        for p in self.fc.parameters():
            p.requires_grad = True

        # print trainable parameters
        print("trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"name: {name}: size: {param.size()}")

    def load_model(self, model_path):
        model = torch.load(model_path)
        # todo
        # the initialize of fc layer maybe enhance
        model["fc.weight"] = torch.rand_like(self.fc.weight)
        model["fc.bias"] = torch.rand_like(self.fc.bias)

        self.load_state_dict(model)

    @staticmethod
    def create_model(model_name, droprate, num_classes):
        model_parameters = {
            "wrn34_10": {
                "depth": 34,
                "widen_factor": 10
            }
        }

        model_param = model_parameters.get(model_name)
        if not model_name:
            raise ValueError("incorrect model name")

        model_param.update({"num_classes": num_classes, "droprate": droprate})

        return getattr(TLWideResNet, model_name)(**model_param)

    @staticmethod
    def wrn34_10(depth, num_classes, widen_factor, droprate):
        return TLWideResNet(
            depth=depth, num_classes=num_classes,
            widen_factor=widen_factor, droprate=droprate
        )


if __name__ == '__main__':
    t: TLResNet = TLWideResNet.create_model("wrn34_10", 0, 100)
    for p in t.parameters():
        if p.requires_grad:
            print(p)
    t.load_model("./trained_models/cifar100_wrn34_10-best")

    test = wrn.wrn34_10(num_classes=100)
    test.load_state_dict(torch.load("./trained_models/cifar100_wrn34_10-best"))

    count = 0
    for a, b in zip(t.parameters(), test.parameters()):
        if not torch.equal(a, b):
            count += 1
    print(count)