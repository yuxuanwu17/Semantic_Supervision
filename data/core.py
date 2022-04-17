from ast import Sub
import pickle 
from pathlib import Path

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from data.utils import LabelDatasetManager

class Cifar100DatasetManagerCore:
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

        self.load_label_meta()

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
            'val': None
        }

        # label dataset
        self.label_dataset = {
            'train': None,
            'val': None
        }

    def load_label_meta(self):
        ''' Get original label information '''
        metadata_path = Path(self.cache_dir).joinpath('cifar-100-python', 'meta')
        assert(metadata_path.exists())

        with metadata_path.open(mode='rb') as f:
            metadata = pickle.load(f, encoding='bytes')
        self.classes = tuple(
            [ele.decode('utf-8') for ele in metadata[b'fine_label_names']]
        )
    def init_label_config(self):
        raise NotImplementedError

    def gen_input_dataset(self):
        raise NotImplementedError
    
    def gen_label_dataset(self):
        # create label dataset manager
        self.label_dataset_manager = LabelDatasetManager(
            cache_dir=self.cache_dir, label_data_args=self.label_data_args,
            train_classes=self.train_class_label, val_classes=self.val_class_label
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