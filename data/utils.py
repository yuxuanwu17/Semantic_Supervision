import numpy as np
import hashlib
import json
from pathlib import Path
from datasets import DatasetDict, load_dataset, load_from_disk, ClassLabel
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, IterableDataset
from collections import defaultdict

class LabelDataset(IterableDataset):
    '''
    Iterable dataset that infinitely interates over randomly sampled labels.
    returns tokenized outputs
        input:
            dataset(DatasetDict)
            class_label(ClassLabel object)
            label_to_idx(dict)
            num_description(int) default 1
                how many descriptions are used for one class in each batch
            multi_description_aggregation(str)
                (only valid when num_description is larger than 1)
                'concat', 'average', 'max'
                how to aggregate multiple description into one embedding
               

    '''
    def __init__(self, dataset: DatasetDict, class_label: ClassLabel, label_to_idx: dict, num_description: int=1):
        super().__init__()
        self.label_to_idx = label_to_idx
        self.class_label = class_label
        self.dataset = dataset
        self.num_classes = class_label.num_classes
        self.num_description = num_description
    
    def __next__(self):
        '''
            For each batch, randomly generate a label description in the format that can be fed into bert model (followed by a forward by label model).
        '''
        bert_input = defaultdict(list)
        for i in range(self.num_classes):
            label = self.class_label.int2str(i)
            choice_list = np.random.choice(self.label_to_idx[label], size=self.num_description, replace=False)
            
            # key: attention_mask, input_ids, token_type_ids
            for choice_item in choice_list:
                for k, v in self.dataset[int(choice_item)].items():
                    bert_input[k].append(v)

        # attention_mask, input_ids, token_type_ids: (num_description * num_classes)
        return {k: torch.stack(v) for k, v in bert_input.items()}
    
    def __iter__(self):
        return self

class LabelDatasetManager:
    ''''
        input:
            cache_dir(str)
            label_data_args(dict):
                label_tokenizer,
                train_label_json,
                val_label_json
            train_classes(ClassLabel)
            val_classes(ClassLabel)
            num_description(int): default 1
            multi_description_aggregation(str):
    '''
    def __init__(
        self,
        cache_dir: str,
        label_data_args: dict,
        train_classes: ClassLabel, val_classes: ClassLabel,
        num_description: int=1, multi_description_aggregation: str='concat'
    ):
        self.cache_dir = cache_dir
        self.label_tokenizer = label_data_args['label_tokenizer']
        self.train_label_json = label_data_args['train_label_json']
        self.val_label_json = label_data_args['val_label_json']
        self.label_max_len = label_data_args['label_max_len']
        self.train_classes = train_classes
        self.val_classes = val_classes
        self.num_description = num_description
        self.multi_description_aggregation = multi_description_aggregation

        assert Path(self.train_label_json).is_file()
        assert Path(self.cache_dir).is_dir()
        if self.val_label_json is not None:
            assert Path(self.val_label_json).is_file()
        else:
            self.val_label_json = self.train_label_json
        
        # change to absolute route
        self.train_label_json = str(Path(self.train_label_json).absolute())
        self.val_label_json = str(Path(self.val_label_json).absolute())

        # generate identifier for label manager
        self.train_label_hash = self.hash_func(
            (self.label_tokenizer, self.train_label_json, self.label_max_len)
        )
        self.val_label_hash = self.hash_func(
            (self.label_tokenizer, self.val_label_json, self.label_max_len)
        )

        # use the identifier as cache path
        self.train_label_cache_path = str(
            Path(self.cache_dir).joinpath('lab_' + self.train_label_hash)
        )
        self.val_label_cache_path = str(
            Path(self.cache_dir).joinpath('lab_' + self.val_label_hash)
        )

        self.label_dataset = {
            'train': None, 'val': None
        }

    def hash_func(self, x):
        return hashlib.md5(json.dumps(x, sort_keys=True).encode("utf-8")).hexdigest()


    def prepare_label_dataset(self):
        '''
            Prepare the label data and save to disk. 
            When the label data has already been processed and saved, this function will not redo the processing.
        '''
        if (not Path(self.train_label_cache_path).exists()):
            tokenizer = AutoTokenizer.from_pretrained(self.label_tokenizer)
            train_label_dataset = load_dataset(
                'json', data_files=self.train_label_json, split='train'
            )
            train_label_dataset = train_label_dataset.map(
                lambda x: tokenizer(
                    x['text'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.label_max_len
                ),
                batched=True
            )
            train_label_dataset.save_to_disk(self.train_label_cache_path)
        if (not Path(self.val_label_cache_path).exists()):
            tokenizer = AutoTokenizer.from_pretrained(self.label_tokenizer)
            val_label_dataset = load_dataset(
                'json', data_files=self.val_label_json, split='train'
            )
            val_label_dataset = val_label_dataset.map(
                lambda x: tokenizer(
                    x['text'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.label_max_len
                ),
                batched=True
            )
            val_label_dataset.save_to_disk(self.val_label_cache_path)

    def gen_one_label_dataset(self, dataset: DatasetDict, class_label: ClassLabel):
        '''
            Generate label dataset for a given dataset with assigned class_label
        '''
        label_to_idx = defaultdict(list)
        for idx, ele in enumerate(dataset):
            label_to_idx[ele['label']].append(idx)
        
        keep_columns = ['input_ids', 'attention_mask']
        if 'token_type_ids' in dataset.column_names:
            keep_columns += ['token_type_ids']
        
        dataset.set_format(type='torch', columns=keep_columns)
        
        return LabelDataset(dataset, class_label, label_to_idx, num_description=self.num_description)


    def gen_label_dataset(self):
        '''
            Generate label datasets for train and validation
        '''
        self.prepare_label_dataset()
        train_dataset = load_from_disk(self.train_label_cache_path)
        val_dataset = load_from_disk(self.val_label_cache_path)

        self.label_dataset['train'] = self.gen_one_label_dataset(train_dataset, self.train_classes)
        self.label_dataset['val'] = self.gen_one_label_dataset(val_dataset, self.val_classes)
