from ast import Sub
import pickle 
from pathlib import Path

from datasets import ClassLabel
from numpy import DataSource
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms
from torchvision.datasets.vision import StandardTransform

from data.utils import LabelDatasetManager
from data.core import Cifar100DatasetManagerCore

CifarHeldoutDataConfig = {
    'val_names': (
        "streetcar",
        "lamp",
        "forest",
        "otter",
        "house",
        "crab",
        "crocodile",
        "orchid",
        "rabbit",
        "man",
    ),
    'test_names': (
        "motorcycle",
        "pine_tree",
        "bottle",
        "trout",
        "chair",
        "butterfly",
        "chimpanzee",
        "orange",
        "leopard",
        "possum",
    )
}

class Cifar100HeldoutDatasetManager(Cifar100DatasetManagerCore):
    def __init__(
        self, 
        general_args, 
        label_model_args, label_data_args,
        train_args
    ):
        super().__init__(
            general_args, 
            label_model_args, label_data_args,
            train_args
        )
        self.init_label_config()


    def init_label_config(self):
        self.heldout_classes = tuple(
                list(CifarHeldoutDataConfig['val_names']) + list(CifarHeldoutDataConfig['test_names'])
            )

        self.train_classes = tuple(
            [x for x in self.classes if x not in self.heldout_classes]
        )

        if self.general_args['run_test']:
            self.val_classes = CifarHeldoutDataConfig['test_names']
        else:
            self.val_classes = CifarHeldoutDataConfig['val_names']
    
        self.all_class_label = ClassLabel(names=self.classes)
        self.train_class_label = ClassLabel(names=self.train_classes)
        self.val_class_label = ClassLabel(names=self.val_classes)


    def gen_input_dataset(self):
        # input dataloader
        # transform will be applied later
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.cache_dir, train=True, download=False
        )
        if self.general_args['run_test']:
            val_dataset = torchvision.datasets.CIFAR100(
                root=self.cache_dir, train=False, download=False
            )
        else:
            val_dataset = torchvision.datasets.CIFAR100(
                root=self.cache_dir, train=True, download=False
            )

        train_class_ids = [
            self.all_class_label.str2int(x) for x in self.classes if x not in self.heldout_classes
        ]
        val_class_ids = [
            self.all_class_label.str2int(x) for x in self.val_classes
        ]

        train_heldout_idx, val_heldout_idx = [], []
        for idx in range(len(train_dataset)):
            _, class_id = train_dataset[idx]
            if class_id in train_class_ids:
                train_heldout_idx.append(idx)
        
        for idx in range(len(val_dataset)):
            _, class_id = val_dataset[idx]
            if class_id in val_class_ids:
                val_heldout_idx.append(idx)
        
        train_dataset = Subset(train_dataset, train_heldout_idx)
        val_dataset = Subset(val_dataset, val_heldout_idx)


        train_target_transform = lambda x: self.train_class_label.str2int(self.all_class_label.int2str(x))
        val_target_transform = lambda x: self.val_class_label.str2int(self.all_class_label.int2str(x))
        
        train_dataset.dataset.transforms = StandardTransform(
            transform=self.train_transform, target_transform=train_target_transform
        )
        val_dataset.dataset.transforms = StandardTransform(
            transform=self.eval_transform, target_transform=val_target_transform
        )

        self.input_dataset['train'] = train_dataset
        self.input_dataset['val'] = val_dataset