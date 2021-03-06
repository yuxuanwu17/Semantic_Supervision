from ast import Sub
import pickle 
from pathlib import Path
import pstats

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from datasets import ClassLabel, load_dataset, concatenate_datasets, load_from_disk
from data.utils import LabelDatasetManager
from data.configs import *
from transformers import AutoTokenizer

class Cifar100DatasetManagerCore:
    def __init__(
        self, general_args, label_model_args, label_data_args, train_args
    ):
        self.general_args = general_args
        self.label_model_args = label_model_args
        self.label_data_args = label_data_args
        self.train_args = train_args

        self.cache_dir = self.general_args['cache_dir']
        self.num_description = label_data_args['num_description'] if 'num_description' in label_data_args else 1
        # self.multi_description_aggregation = label_data_args['multi_description_aggregation'] if 'multi_description_aggregation' in label_data_args else 'concat'
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
            train_classes=self.train_class_label, val_classes=self.val_class_label,
            num_description=self.num_description
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



class AWADataset(ImageFolder):
    def __init__(self, class_label: ClassLabel, *args, **kwargs):
        self.class_label = class_label
        super().__init__(*args, **kwargs)
    
    def find_classes(self, *args, **kwargs):
        super().find_classes(*args, **kwargs)
        classes = list(self.class_label.names)
        cls_to_idx = dict()
        for cls in classes:
            cls_to_idx[cls] = self.class_label.str2int(cls)
        
        return classes, cls_to_idx

class AWADatasetManagerCore:
    def __init__(self, general_args, label_model_args, label_data_args, train_args):
        self.general_args = general_args
        self.label_model_args = label_model_args
        self.label_data_args = label_data_args
        self.train_args = train_args

        # load label meta
        self.classes = AWA_ALL_CLASSES
        self.cache_dir = self.general_args['cache_dir']
        self.num_description = label_data_args['num_description'] if 'num_description' in label_data_args else 1
        self.dataset_dir = Path(self.cache_dir).joinpath("Animals_with_Attributes2")
        self.img_root = self.dataset_dir.joinpath("JPEGImages")

         #  stats
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # transforms
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
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

    def init_label_config(self):
        '''
            initialize train_class_label and val_class_label
        '''
        raise NotImplementedError

    def gen_input_dataset(self):
        raise NotImplementedError

    def gen_label_dataset(self):
        self.label_dataset_manager = LabelDatasetManager(
            cache_dir=self.cache_dir, label_data_args=self.label_data_args,
            train_classes=self.train_class_label,
            val_classes=self.val_class_label,
            num_description=self.num_description
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
        

class NewsgroupsDatasetManagerCore:
    def __init__(
        self, general_args, input_model_args, input_data_args, label_model_args, label_data_args, train_args
    ):
        self.general_args = general_args
        self.input_model_args = input_model_args
        self.input_data_args = input_data_args
        self.label_model_args = label_model_args
        self.label_data_args = label_data_args
        self.train_args = train_args

        # load label meta
        self.classes = NewsgroupsAllClasses
        self.cache_dir = self.general_args['cache_dir']

        self.dataset_cache_path = str(None)
        
        self.num_description = label_data_args['num_description'] if 'num_description' in label_data_args else 1
        # self.multi_description_aggregation = label_data_args['multi_description_aggregation'] if 'multi_description_aggregation' in label_data_args else 'concat'
        
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

    def init_label_config(self):
        '''
            initialize train_class_label and val_class_label
        '''
        raise NotImplementedError

    def _configure_dataset(self, dataset, tokenizer, class_name: str):
        """Add labels to the newsgroups dataset and tokenize it"""
        dataset = dataset.map(
            lambda x: {"labels": len(x["text"]) * [class_name]}, batched=True
        )
        dataset = dataset.map(
            lambda x: tokenizer(
                x["text"],
                truncation=True,
                padding="max_length",
                max_length=self.input_data_args['input_max_len'],
            ),
            batched=True,
        )
        return dataset

    def prepare_input_dataset(self):
        if Path(self.dataset_cache_path).exists():
            return
        # input dataloader
        tokenizer = AutoTokenizer.from_pretrained(self.input_data_args['input_tokenizer'])
        dataset_list = []
        for c in self.classes:
            dataset_list.append(
                self._configure_dataset(
                    load_dataset(
                        "newsgroup", f"{self.general_args['variant']}_{c}", split="train"
                    ),
                    tokenizer,
                    class_name=c,
                )
            )
        combined_datasets = concatenate_datasets(dataset_list)
        combined_datasets.save_to_disk(self.dataset_cache_path)

    def gen_input_dataset(self):
        raise NotImplementedError
    
    def gen_label_dataset(self):
        # create label dataset manager
        self.label_dataset_manager = LabelDatasetManager(
            cache_dir=self.cache_dir, label_data_args=self.label_data_args,
            train_classes=self.train_class_label, val_classes=self.val_class_label,
            num_description=self.num_description
        )
        # generate label dataset
        self.label_dataset_manager.gen_label_dataset()
        self.label_dataset = self.label_dataset_manager.label_dataset
    
    def gen_dataset(self):
        self.gen_label_dataset()

        self.dataset_hash = self.label_dataset_manager.hash_func(
            (
                "newsgroups",
                sorted(list(self.classes)),
                self.input_data_args['input_max_len'],
                self.general_args['split_seed'],
                self.general_args['test_size'],
                self.general_args['val_size'],
                self.general_args['variant'],
            )
        )
        self.dataset_cache_path = str(
            Path(self.cache_dir).joinpath("ng_" + self.dataset_hash)
        )

        self.gen_input_dataset()
        
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