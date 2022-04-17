import torchvision

cache_dir = './data_cache'

torchvision.datasets.CIFAR100(
    root=cache_dir, train=True, download=True
)