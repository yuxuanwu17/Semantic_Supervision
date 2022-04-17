from ast import Sub
import pickle 
from pathlib import Path

from datasets import ClassLabel
from numpy import DataSource
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split

from data.utils import LabelDatasetManager
from data.core import Cifar100DatasetManagerCore

class Cifar100DatasetManager(Cifar100DatasetManagerCore):
    '''
        input:
            general_args(dict): 
                cache_dir: str
                split_seed: int
                val_size: float
                num_workers: int
                run_test: bool

            label_model_args(dict):
                label_model: str

            label_data_args(dict):
                label_tokenizer: str
                train_label_json: str
                val_label_json: str
                label_max_len: int

            train_args(dict):
                num_epochs: int
                log_every_n_steps: int
                train_batch_size: int
                val_batch_size: int
                pretrained_model: bool
                tune_label_model: bool
    '''
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
        self.train_classes = self.classes
        self.val_classes = self.classes
        
        self.train_class_label = ClassLabel(names=self.train_classes)
        self.val_class_label = ClassLabel(names=self.val_classes)

    def gen_input_dataset(self):
        # input dataloader
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.cache_dir, train=True, download=False, transform=self.train_transform
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=self.cache_dir, train=True, download=False, transform=self.eval_transform
        )

        if self.general_args['run_test']:
            val_dataset = torchvision.datasets.CIFAR100(
                root=self.cache_dir, train=False, download=False, transform=self.eval_transform
            )
        
        train_idx, val_idx = train_test_split(
            list(range(len(train_dataset))),
            test_size=self.general_args['val_size'],
            random_state=self.general_args['split_seed'],
            shuffle=True,
            stratify=train_dataset.targets
        )

        self.input_dataset['train'] = Subset(train_dataset, train_idx)
        self.input_dataset['val'] = Subset(val_dataset, val_idx)
