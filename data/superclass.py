from ast import Sub
import pickle 
from pathlib import Path

from datasets import ClassLabel
from numpy import DataSource
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets.vision import StandardTransform
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split

from data.utils import LabelDatasetManager
from data.core import Cifar100DatasetManagerCore
from data.configs import *

class Cifar100DSuperClassDatasetManager(Cifar100DatasetManagerCore):
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
        self.train_level = self.general_args['train_level']
        self.val_level = self.general_args['val_level']

        self.classes_to_superclass = {
            x: k for k, v in CifarSuperClass2Classes.items() for x in v
        }
        self.init_label_config()

    def init_label_config(self):
        '''
            fine - class
            coarse - superclass
        '''
        self.val_superclasses = CifarSuperClassConfig["test_superclasses"] if self.general_args['run_test'] \
                                else CifarSuperClassConfig["val_superclasses"]

        self.train_classes = self.classes if self.train_level == 'fine' \
                                                else CifarSuperClassConfig["superclasses"]
        self.val_classes = self.classes if self.val_level == 'fine' \
                                else self.val_superclasses

        self.all_class_label = ClassLabel(names=self.classes)
        self.train_class_label = ClassLabel(names=self.train_classes)
        self.val_class_label = ClassLabel(names=self.val_classes)

    def gen_input_dataset(self):
        def train_target_transform(x):
            if self.train_level == 'fine':
                return x
            return self.train_class_label.str2int(
                        self.classes_to_superclass(self.all_class_label.int2str(x)))
    
        def val_target_transform(x):
            if self.val_level == 'fine':
                return x
            return self.val_class_label.str2int(
                        self.classes_to_superclass[self.all_class_label.int2str(x)])
        

        train_dataset = torchvision.datasets.CIFAR100(
            root=self.cache_dir, train=True, download=False, 
            transform=self.train_transform,
            target_transform=train_target_transform
        )

        train_idx, val_idx = train_test_split(
            list(range(len(train_dataset))),
            test_size=self.general_args['val_size'],
            random_state=self.general_args['split_seed'],
            shuffle=True,
            stratify=train_dataset.targets
        )

        self.input_dataset['train'] = Subset(train_dataset, train_idx)


        if self.general_args['run_test']:
            test_dataset =  torchvision.datasets.CIFAR100(
                                root=self.cache_dir, train=True, download=False, 
                                transform=self.eval_transform
                            )
            if self.val_level == 'coarse':
                test_idx = []
                test_class_ids = [
                    self.all_class_label.str2int(x) for x in self.classes \
                        if self.classes_to_superclass[x] in self.val_superclasses
                ]
                for idx in range(len(test_dataset)):
                    _, class_id = test_dataset[idx]
                    if class_id in test_class_ids:
                        test_idx.append(idx)
                test_dataset = Subset(test_dataset, test_idx)
            
            test_dataset.dataset.target_transform = val_target_transform
            
            self.input_dataset['val'] = test_dataset
        
        else:
            val_dataset = torchvision.datasets.CIFAR100(
                root=self.cache_dir, train=True, download=False, 
                transform=self.eval_transform
            )
            # furthur filter val dataset to remove instances whose superclass is not in the val superclass
            if self.val_level == 'coarse':
                val_final_idx = []
                val_class_ids = [
                    self.all_class_label.str2int(x) for x in self.classes \
                        if self.classes_to_superclass[x] in self.val_superclasses
                ]
                for idx in val_idx:
                    _, class_id = train_dataset[idx]
                    if class_id in val_class_ids:
                        val_final_idx.append(idx)
                val_idx = val_final_idx
            
            val_dataset = Subset(val_dataset, val_idx)
            val_dataset.dataset.target_transform = val_target_transform
            
            self.input_dataset['val'] = val_dataset


    
        