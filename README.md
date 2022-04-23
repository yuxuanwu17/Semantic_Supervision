# Semantic_Supervision

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
python3 main.py --config [your_path_to_configuration_file]
```

For example, to run SemSup generalization to unseen task on CIFAR dataset, run 

```shell
python3 main.py --config ./config/cifar/scene1.json
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
            "task": input encoding model, "cifar" and "awa" will use ResNetSemSup, "ng" will use Bert model (todo)
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

