from ast import Sub
import pickle 
from pathlib import Path

from datasets import ClassLabel, DatasetDict
from numpy import DataSource
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms
from torchvision.datasets.vision import StandardTransform

from data.utils import LabelDatasetManager
from data.core import *
from data.configs import *

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
        
        train_dataset.dataset.transform = self.train_transform
        train_dataset.dataset.target_transform = train_target_transform
        train_dataset.dataset.transforms = StandardTransform(
            transform=self.train_transform, target_transform=train_target_transform
        )

        val_dataset.dataset.transform = self.eval_transform
        val_dataset.dataset.target_transform = val_target_transform
        val_dataset.dataset.transforms = StandardTransform(
            transform=self.eval_transform, target_transform=val_target_transform
        )

        self.input_dataset['train'] = train_dataset
        self.input_dataset['val'] = val_dataset


class AWAHeldoutDatasetManager(AWADatasetManagerCore):
    def __init__(self, general_args, label_model_args, label_data_args, train_args):
        super().__init__(general_args, label_model_args, label_data_args, train_args)
        self.init_label_config()

    def init_label_config(self):
        self.train_classes = AWA_TRAIN_CLASSES
        self.val_classes = AWA_TEST_CLASSES if self.general_args['run_test'] else AWA_VAL_CLASSES

        self.train_class_label = ClassLabel(names=self.train_classes)
        self.val_class_label = ClassLabel(names=self.val_classes)

    def gen_input_dataset(self):
        self.input_dataset['train'] = AWADataset(
            class_label=self.train_class_label,
            transform=self.train_transform,
            root=self.img_root
        )

        self.input_dataset['val'] = AWADataset(
            class_label=self.val_class_label,
            transform=self.eval_transform,
            root=self.img_root
        )
        
class NewsgroupsHeldoutDatasetManager(NewsgroupsDatasetManagerCore):
    def __init__(
        self, 
        general_args,
        input_model_args, 
        input_data_args, 
        label_model_args, label_data_args,
        train_args
    ):
        super().__init__(
            general_args,
            input_model_args,
            input_data_args,
            label_model_args, label_data_args,
            train_args
        )
        self.init_label_config()
    
    def init_label_config(self):
        self.heldout_classes = tuple(list(NewsgroupsHeldoutDataConfig['val_names']) + list(NewsgroupsHeldoutDataConfig['test_names']))
        self.train_classes = tuple(
            [x for x in self.classes if x not in self.heldout_classes]
        )
        self.val_classes = (
            NewsgroupsHeldoutDataConfig['val_names']
        )
        self.test_classes = (
            NewsgroupsHeldoutDataConfig['test_names']
        )

        if self.general_args['run_test']:
            self.val_classes = self.test_classes

        self.train_class_label = ClassLabel(names=self.train_classes)
        self.val_class_label = ClassLabel(names=self.val_classes)

    def gen_input_dataset(self):
        self.prepare_input_dataset()
        loaded_dataset = load_from_disk(self.dataset_cache_path)
        dataset = DatasetDict()

        dataset["train"] = loaded_dataset.filter(
            lambda x: x["labels"] in self.train_class_label.names
        )
        dataset["train"] = dataset["train"].map(
            lambda x: {"labels": self.train_class_label.str2int(x["labels"])}
        )
        
        dataset["val"] = loaded_dataset.filter(
            lambda x: x["labels"] in self.val_class_label.names
        ) 
        dataset["val"] = dataset["val"].map(
            lambda x: {"labels": self.val_class_label.str2int(x["labels"])}
        )
        dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
        )

        self.input_dataset['train'] = dataset["train"]
        self.input_dataset['val'] = dataset["val"]