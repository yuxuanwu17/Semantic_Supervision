# Semantic_Supervision

## Code Structure
### Environment preparation
- download.sh - download class description 
- init.py - download dataset 
- setup.py - install required packages

### Code file structure
- **main.py**
- **model** 
    - text.py - NLP task 
    - vision.py - CV task 
    - utils.py
- **data**
    - core.py - core dataloader
    - base.py - scenario 1
    - heldout.py - scenario 2
    - superclass.py - scenario 3
    - configs.py - meta class information and split
    - utils.py
- **config** 
    - awa
    - cifar
    - newsgroups

### Code design
- **main.py** - main training entrance. Detailed instructions are provided below.
- **model** - contains models for the CV task and the NLP task. Each model feeds inputs to the ResNet/Bert model and feeds label descriptions to the Bert model. 
- **data** - contains customized data loaders. The data loader conducts data loading, preprocessing and transforming for input data and label descriptions. Each scenario uses a different customized data loader, which extends from the base data loader in core.py.
- **config** - contains training configurations for different datasets and different scenarios


## Setup


Create python 3.7 virtual environment and in the current folder, run
```shell
sh setup.sh
sh download.sh
python3 init.py
mkdir ckpt
mkdir data_cache
```

## Run
### Configuration
Configuration is specified in json format. Detailed interpretation of each field is explained later. Configuration files for CIFAR can be found in ./config/cifar, files for AWA can be foundin ./config/AWA. 

### Train and validation
```shell
python3 main.py --config [your_path_to_configuration_file] --save_dir [repeat_exp_id]
```

For example, to run SemSup generalization to unseen task on CIFAR dataset, run 

```shell
python3 main.py --config ./config/cifar/scene1.json --save_dir rep1
```

### Test
```shell
python3 main.py --config [your_path_fo_configuration_file]] --run_test True --ckpt_path [your_path_to_checkpoint]
```

### Configuration specification

```yaml
{
    "meta": 
        {
            "dataset": dataset, one of "cifar", "awa" and "ng"
            "task": input encoding model, "cifar" and "awa" will use ResNetSemSup, "ng" will use BertSemSup
            "scene": one of "base", "heldout", and "superclass"
            "ckpt_dir": checkpoint directory
            "name": case name for checkpoint save
        },
    "general_args":
        {
            "cache_dir": dataset and running time cache directory
            "split_seed": seed in train/validation/test division
            "val_size": validation size
            "num_workers": 0
        },
    "input_model_args": # only applicable to newsgroups
        {
            "input_model": "prajjwal1/bert-small" 
        },
    "input_data_args": # only applicable to newsgroups
        {
            "input_tokenizer": "prajjwal1/bert-small",
            "input_max_len": 512
        },
    "label_model_args":
        {
            "label_model": "prajjwal1/bert-small"
        },
    "label_data_args":
        {
            "label_tokenizer": "prajjwal1/bert-small",
            "train_label_json": path to train label description json file,
            "val_label_json": path to validation/test label description json file,
            "label_max_len": 128
        },
    "train_args":
        {
            "num_epochs": total training epochs,
            "train_batch_size": training batch size,
            "val_batch_size": validation/test batch size (note: small batch size on test gives more stable results),
            "pretrained_model": whether to tune input encoding model,
            "tune_label_model": whether to tune label encoding model
        },
    "optimizer_args":
        {
            "lr": AdamW learning rate,
            "adam_epsilon": Adam epos,
            "weight_decay": AdamW weight decay
        }
}
```
### Reference
    Austin W Hanjie, Ameet Deshpande, and Karthik Narasimhan. 2022. Semantic Supervision: Enabling
    Generalization over Output Spaces. arXiv preprint arXiv:2202.13100 (2022).
    https://github.com/princeton-nlp/semsup
