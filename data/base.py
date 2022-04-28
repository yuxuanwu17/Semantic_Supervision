import pickle 
from pathlib import Path

import datasets
from datasets import ClassLabel
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from sklearn.model_selection import train_test_split

from data.core import *

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
        
        train_idx, val_idx = train_test_split(
            list(range(len(train_dataset))),
            test_size=self.general_args['val_size'],
            random_state=self.general_args['split_seed'],
            shuffle=True,
            stratify=train_dataset.targets
        )

        self.input_dataset['train'] = Subset(train_dataset, train_idx)
        self.input_dataset['val'] = Subset(val_dataset, val_idx)

        if self.general_args['run_test']:
            self.input_dataset['val'] = torchvision.datasets.CIFAR100(
                root=self.cache_dir, train=False, download=False, transform=self.eval_transform
            )

class AWADatasetManager(AWADatasetManagerCore):
    def __init__(self, general_args, label_model_args, label_data_args, train_args):
        super().__init__(general_args, label_model_args, label_data_args, train_args)
        self.init_label_config()
    
    def init_label_config(self):
        self.train_classes = self.classes
        self.val_classes = self.classes
        
        self.train_class_label = ClassLabel(names=self.train_classes)
        self.val_class_label = ClassLabel(names=self.val_classes)

    def gen_input_dataset(self):
        train_dataset = AWADataset(
            class_label=self.train_class_label,
            transform=self.train_transform,
            root=self.img_root
        )

        val_dataset = AWADataset(
            class_label=self.val_class_label,
            transform=self.eval_transform,
            root=self.img_root
        )

        train_idx, test_idx = train_test_split(
            list(range(len(train_dataset))),
            test_size=self.general_args['val_size'],
            random_state=self.general_args['split_seed'],
            shuffle=True,
            stratify=train_dataset.targets
        )

        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=self.general_args['val_size'],
            random_state=self.general_args['split_seed'],
            shuffle=True,
            stratify=[train_dataset.targets[idx] for idx in train_idx]
        )

        self.input_dataset['train'] = Subset(train_dataset, train_idx)
        # notice that once the seed is fixed, each time the train_test_split generates the same results
        self.input_dataset['val'] = Subset(val_dataset, test_idx) if self.general_args['run_test'] \
                                        else Subset(val_dataset, val_idx)

class NewsgroupsDatasetManager(NewsgroupsDatasetManagerCore):
    '''
        input:
            general_args(dict): 
                cache_dir: str
                split_seed: int
                val_size: float
                num_workers: int
                run_test: bool

            input_model_args(dict):
                input_model: str

            input_data_args(dict):
                input_tokenizer: str
                input_max_len: int 
            
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
        self.train_classes = self.classes
        self.val_classes = self.classes
        
        self.train_class_label = ClassLabel(names=self.train_classes)
        self.val_class_label = ClassLabel(names=self.val_classes)

    def gen_input_dataset(self):
        self.prepare_input_dataset()
        dataset = load_from_disk(self.dataset_cache_path)
        
        # make train-test split
        train_test = dataset.train_test_split(
            test_size=self.general_args["test_size"], seed=self.general_args["split_seed"]
        )

        # split train set into train-val
        train_val = train_test["train"].train_test_split(
            test_size=self.general_args["val_size"], seed=self.general_args["split_seed"]
        )

        self.input_dataset['train'] = train_val["train"]
        self.input_dataset['val'] = train_test["test"] if self.general_args['run_test'] else train_val["test"]