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

class Cifar100DatasetManager:
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
        self.general_args = general_args
        self.label_model_args = label_model_args
        self.label_data_args = label_data_args
        self.train_args = train_args

        self.cache_dir = self.general_args['cache_dir']

        # label dataloader

        # input transform
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]

        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        self.eval_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        )

        # input dataset
        self.input_dataset = {
            'train': None, 
            'val': None,
            'test': None
        }

        # label dataset
        self.label_dataset = {
            'train': None,
            'val': None,
            'test': None
        }

    def gen_input_dataset(self):
        # input dataloader
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.cache_dir, train=True, download=False, transform=self.train_transform
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=self.cache_dir, train=True, download=False, transform=self.eval_transform
        )

        if self.general_args['run_test']:
            test_dataset = torchvision.datasets.CIFAR100(
                root=self.cache_dir, train=False, download=False, transform=self.eval_transform
            )
            self.input_dataset['test'] = test_dataset
        
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
            self.input_dataset['val'] = self.input_dataset['test']
    
    def gen_label_dataset(self):
        ''' Get label information '''
        metadata_path = Path(self.cache_dir).joinpath('cifar-100-python', 'meta')
        assert(metadata_path.exists())

        with metadata_path.open(mode='rb') as f:
            metadata = pickle.load(f, encoding='bytes')
        self.classes = tuple(
            [ele.decode('utf-8') for ele in metadata[b'fine_label_names']]
        )
        self.train_classes = self.classes
        self.val_classes = self.classes
        
        train_class_label = ClassLabel(names=self.train_classes)
        val_class_label = ClassLabel(names=self.val_classes)

        # create label dataset manager
        self.label_dataset_manager = LabelDatasetManager(
            cache_dir=self.cache_dir, label_data_args=self.label_data_args,
            train_classes=train_class_label, val_classes=val_class_label
        )
        # generate label dataset
        self.label_dataset_manager.gen_label_dataset()
        self.label_dataset = self.label_dataset_manager.label_dataset
    
    def gen_dataset(self):
        self.gen_input_dataset()
        self.gen_label_dataset()

        # now we have everthing is in self.input_dataset and self.label_dataset
        self.train_dataloader = {
            'input_loader': DataLoader(
                self.input_dataset['train'],
                batch_size=self.train_args['train_batch_size'],
                num_workers=self.general_args['num_workers'],
                shuffle=True
            ),
            'label_loader': DataLoader(
                self.label_dataset['train'], num_workers=1
            )
        }

        self.val_dataloader = {
            'input_loader': DataLoader(
                self.input_dataset['val'],
                batch_size=self.train_args['val_batch_size'],
                num_workers=self.general_args['num_workers'],
                shuffle=False
            ),
            'label_loader': DataLoader(
                self.label_dataset['val'], num_workers=1
            )
        }
