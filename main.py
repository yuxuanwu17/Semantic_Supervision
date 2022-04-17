import json
import os

from data.base import Cifar100DatasetManager
from model.vision import ResNetSemSup
from torch.optim import AdamW
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from tqdm import tqdm
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print("Device: ", device)

def train(model, optimizer, criterion, input_loader, label_loader, scheduler, epoch, num_epoch):
    model.train()
    batch_bar = tqdm(total=len(input_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    total_loss = 0.0
    start = time.time()

    pred_list, true_list = [], []
    
    for batch_idx, x in enumerate(input_loader):
        optimizer.zero_grad()
        label_batch = next(label_loader)
        input_batch, y = x
        input_batch = input_batch.to(device)
        y = y.to(device)
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

def validate(model, input_loader, label_loader, criterion):
    model.eval()
    start = time.time()
    batch_bar = tqdm(total=len(input_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')
    with torch.no_grad():
        total_loss = 0
        pred_list, true_list = [], []
        for batch_idx, x in enumerate(input_loader):
            label_batch = next(label_loader)
            input_batch, y = x
            input_batch = input_batch.to(device)
            y = y.to(device)
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



if __name__ == '__main__':
    with open('./config/cifar/scene1.json', 'r') as f:
        config = json.load(f)

    general_args = config['general_args']
    label_model_args = config['label_model_args']
    label_data_args = config['label_data_args']
    train_args = config['train_args']
    optimizer_args = config['optimizer_args']

    dataset_manager = Cifar100DatasetManager(
                        general_args, label_data_args, label_data_args, train_args)
    dataset_manager.gen_dataset()
    train_data_loader = dataset_manager.train_dataloader
    train_input_loader, train_label_loader = train_data_loader['input_loader'], \
                                             iter(train_data_loader['label_loader'])

    val_data_loader = dataset_manager.val_dataloader
    val_input_loader, val_label_loader = val_data_loader['input_loader'], \
                                         iter(val_data_loader['label_loader'])

    model = ResNetSemSup(train_args, label_model_args, 'cifar').to(device)

    num_epochs = train_args['num_epochs']
    lr = optimizer_args['lr']
    eps = optimizer_args['adam_epsilon']
    weight_decay = optimizer_args['weight_decay']

    optimizer = AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 1

    for epoch in range(num_epochs):
        train(model, optimizer, criterion, train_input_loader, train_label_loader, None, epoch, num_epochs)
        validate(model, val_input_loader, val_label_loader, criterion)




