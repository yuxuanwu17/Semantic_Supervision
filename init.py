import torchvision
import os
from pathlib import Path
from torchvision.datasets.utils import download_and_extract_archive

cache_dir = './data_cache'
cifar_path = Path(cache_dir).joinpath('cifar-100-python')
awa_path = Path(cache_dir).joinpath("Animals_with_Attributes2")

if not cifar_path.exists():
    torchvision.datasets.CIFAR100(
        root=cache_dir, train=True, download=True
    )

if not awa_path.exists():
    data_url = "http://cvml.ist.ac.at/AwA2/AwA2-data.zip"
    download_and_extract_archive(
                url=data_url,
                download_root=cache_dir
            )