"""`learning without forgetting` in transfer learning

training steps:
    1. PGD-7 adversarial training on teacher model(e.g. cifar100)
    2. initialize student model from robust teacher model(with reshaped fc layer)
    3. calculate feature representations of **student dataset**(e.g. cifar10 dataset) with initialized student model
    4. store feature representations in memory
        - custom defined Dataloader could be used
    5. use loss: f(x, y_hat) +
                 λ * torch.mean(torch.norm(stored feature representations - running feature representations, p=1, dim=1))
       to train student model with benign student dataset(e.g. cifar10 dataset)
        - in warm-start step only train fully connect(last) layer
        - after warm-start step, train whole model

questions:
    1. `weight decay`: 论文代码中采用 0.0002, 我们一直用的是 0.0005
    2. `epoch`: 论文代码训练了 20000 个 steps, 相当于 51.15 个 epochs, 我们在之前的训练中一直采用 100 个 epochs
    3. `learning rate`: 论文代码采用 0.001 从头至尾, 我们之前的训练中使用初始 lr = 0.1, learning rate decay = 0.2,
                        momentum = [40, 70, 90], 并且我们在第一个 epoch 使用了 warm-up
    4. `warm-start`: 论文代码前 10000 个 steps 只训练 fully connect layer, 之后训练整个模型，我们应该如何设置
"""


