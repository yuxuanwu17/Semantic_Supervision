from argparse import ArgumentParser
import json
import os
from pathlib import Path

from yaml import parse

from data.base import *
from data.heldout import *
from data.superclass import *
from model.vision import ResNetSemSup
from model.text import BertSemSup
from torch.optim import AdamW
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import accuracy_score
import wandb
from tqdm import tqdm
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print("Device: ", device)

DatasetManagerMapping = {
    'cifar': {
        'base': Cifar100DatasetManager,
        'heldout': Cifar100HeldoutDatasetManager,
        'superclass': Cifar100DSuperClassDatasetManager
    },
    'awa': {
        'base': AWADatasetManager,
        'heldout': AWAHeldoutDatasetManager
    },
    'newsgroups': {
        'base': NewsgroupsDatasetManager,
        'heldout': NewsgroupsHeldoutDatasetManager,
        # 'superclass': NewsgroupsSuperClassDatasetManager
    }
}

ModelMapping = {
    'cifar': ResNetSemSup,
    'awa': ResNetSemSup,
    'newsgroups': BertSemSup
}


def train(model, optimizer, criterion, input_loader, label_loader, scheduler, epoch, num_epoch):
    model.train()
    batch_bar = tqdm(total=len(input_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    total_loss = 0.0
    start = time.time()

    pred_list, true_list = [], []

    for batch_idx, x in enumerate(input_loader):
        optimizer.zero_grad()
        if len(x) == 2:
            input_batch, y = x
            input_batch = input_batch.to(device)
        else:
            input_batch = {'attention_mask':x['attention_mask'], 
                            'input_ids':x['input_ids'], 
                            'token_type_ids':x['token_type_ids']}
            input_batch = {k: v.to(device) for k, v in input_batch.items()}
            y = x['labels']
        y = y.to(device)
        label_batch = next(label_loader)
        label_batch = {k: v.to(device) for k, v in label_batch.items()}
        logits = model((input_batch, label_batch))
        pred = torch.argmax(logits, axis=1)

        pred_list.extend(pred.tolist())
        true_list.extend(y.tolist())

        loss = criterion(logits, y)
        total_loss += float(loss)

        batch_bar.set_postfix(
            loss="{:04f}".format(float(total_loss / (batch_idx + 1))),
            lr="{:04f}".format(float(optimizer.param_groups[0]['lr']))
        )

        loss.backward()
        optimizer.step()
        batch_bar.update()

        if scheduler is not None:
            scheduler.step()

    end = time.time()
    batch_bar.close()
    total_loss /= len(input_loader)
    accuracy = accuracy_score(true_list, pred_list)

    print('Epoch: {}/{}: Train Loss: {:.04f}, Accuracy: {:.04f}, Learning Rate: {:.04f}, Total time: {:.02f}'.format(
        epoch + 1, num_epoch, total_loss, accuracy, optimizer.param_groups[0]['lr'], end - start
    ))

    return total_loss, accuracy, optimizer.param_groups[0]['lr']


def validate(model, input_loader, label_loader, criterion):
    model.eval()
    start = time.time()
    batch_bar = tqdm(total=len(input_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')
    with torch.no_grad():
        total_loss = 0
        pred_list, true_list = [], []
        for batch_idx, x in enumerate(input_loader):
            if len(x) == 2:
                input_batch, y = x
                input_batch = input_batch.to(device)
            else:
                input_batch = {'attention_mask':x['attention_mask'], 
                                'input_ids':x['input_ids'], 
                                'token_type_ids':x['token_type_ids']}
                input_batch = {k: v.to(device) for k, v in input_batch.items()}
                y = x['labels']
            y = y.to(device)
            label_batch = next(label_loader)
            label_batch = {k: v.to(device) for k, v in label_batch.items()}
            logits = model((input_batch, label_batch))
            pred = torch.argmax(logits, axis=1)
            pred_list.extend(pred.tolist())
            true_list.extend(y.tolist())
            loss = criterion(logits, y)
            total_loss += float(loss)
            batch_bar.set_postfix(
                loss="{:04f}".format(float(total_loss) / (batch_idx + 1))
            )
            batch_bar.update()

    val_acc = accuracy_score(true_list, pred_list)
    batch_bar.close()
    end = time.time()

    total_loss /= len(input_loader)
    print('Validation loss: {:.04f}, accuracy: {:.04f}, Time: {:.02f}'.format(
        total_loss, val_acc, end - start
    ))

    return total_loss, val_acc


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for training')
    parser.add_argument('--run_test', type=bool, default=False,
                        help='True for test, False for train')
    parser.add_argument('--ckpt_path', type=str, default='',
                        help='Path of checkpoint used for test, only effective when run_test is True')
    parser.add_argument('--config', type=str, default='./config/cifar/scene1.json',
                        help='configuration file to use')
    parser.add_argument('--save_dir', type=str, default='',
                        help='save the ckpt path with specific names')
    args = parser.parse_args()

    config_file = args.config
    with open(config_file, 'r') as f:
        config = json.load(f)

    general_args = config['general_args']
    general_args['run_test'] = args.run_test
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    input_model_args = config['input_model_args'] if 'input_model_args' in config else None
    input_data_args = config['input_data_args'] if 'input_data_args' in config else None
    label_model_args = config['label_model_args']
    label_data_args = config['label_data_args']
    train_args = config['train_args']
    optimizer_args = config['optimizer_args']

    dataset = config['meta']['dataset']
    scene = config['meta']['scene']
    task = config["meta"]["task"]
    dataset_manager_class = DatasetManagerMapping[dataset][scene]
    score_function_type = 'base'

    score_function_args = config['score_function_args'] if 'score_function_args' in config else None

    model_class = ModelMapping[task]

    if dataset == 'newsgroups':
        dataset_manager = dataset_manager_class(
            general_args, input_model_args, input_data_args, label_model_args, label_data_args, train_args)
    else:
        dataset_manager = dataset_manager_class(
            general_args, label_model_args, label_data_args, train_args)
    dataset_manager.gen_dataset()
    train_data_loader = dataset_manager.train_dataloader
    train_input_loader, train_label_loader = train_data_loader['input_loader'], \
                                             iter(train_data_loader['label_loader'])

    val_data_loader = dataset_manager.val_dataloader
    val_input_loader, val_label_loader = val_data_loader['input_loader'], \
                                         iter(val_data_loader['label_loader'])
    if task == 'newsgroups':
        model = model_class(train_args, input_model_args, label_model_args, score_function_args, task).to(device)
    else:
        model = model_class(train_args, label_model_args, score_function_args, task).to(device)
    print('Model:')
    print(model)

    num_epochs = train_args['num_epochs']
    lr = optimizer_args['lr']
    eps = optimizer_args['adam_epsilon']
    weight_decay = optimizer_args['weight_decay']

    optimizer = AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # set wandb
    model_id = config["meta"]["ckpt_dir"] + config["meta"]["name"] + "_" + args.save_dir
    if args.run_test:
        model_id += "_test"
    wandb.init(project="Semantic_Supervision_Impl_Repeats", entity="saltedfish", name=model_id)
    wandb.config = {
        "learning_rate": lr,
        "epochs": num_epochs,
    }

    if args.run_test:
        assert Path(args.ckpt_path).is_file()

        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        val_loss = checkpoint['val_loss']
        val_acc = checkpoint['val_acc']
        print('--------------Test-------------')
        print('Validation loss: {}, validation acc: {}'.format(val_loss, val_acc))
        test_loss, test_acc = validate(
            model, val_input_loader, val_label_loader, criterion
        )
        print('Test loss: {}, test acc: {}'.format(test_loss, test_acc))
        wandb.log({"test loss": test_loss, "test_acc": test_acc, "val loss": val_loss, "val acc": val_acc})

    else:
        ckpt_path = config["meta"]["ckpt_dir"] + config["meta"]["name"] + "_" + args.save_dir
        for epoch in range(num_epochs):
            train_loss, train_acc, train_lr = train(model, optimizer, criterion, train_input_loader, train_label_loader,
                                                    None, epoch, num_epochs)
            val_loss, val_acc = validate(model, val_input_loader, val_label_loader, criterion)

            wandb.log({"epoch": epoch, "train loss:": train_loss, "train acc": train_acc, "val loss:": val_loss,
                       "val acc": val_acc, "lr": train_lr})

            cur_ckpt_path = ckpt_path + '_{}.pt'.format(epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, cur_ckpt_path)





