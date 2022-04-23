from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.utils import get_label_model



class ResNetSemSup(nn.Module):
    '''
        input:
            train_args:
                pretrained_model: bool
                tune_label_model: bool
            label_model_args:
                label_model: str
    '''
    def __init__(self, train_args: dict, label_model_args: dict, score_function_args: dict, task: str):
        super().__init__()
        self.train_args = train_args
        self.label_model_args = label_model_args
        self.task = task
        self.input_model = models.resnet18(pretrained=self.train_args['pretrained_model'])
        self.label_model = get_label_model(label_model=self.label_model_args['label_model'], 
                                           pretrained=self.train_args['tune_label_model'])
        

        # label model args
        self.num_description = label_model_args['num_description'] if 'num_description' in label_model_args else 1
        self.multi_description_aggregation = label_model_args['multi_description_aggregation'] if 'multi_description_aggregation' in label_model_args else 'concat'
        assert self.multi_description_aggregation in ['concat', 'mean']

        if self.multi_description_aggregation == 'concat':
            self.label_model_hidden = self.label_model.config.hidden_size * self.num_description
        else:
            self.label_model_hidden = self.label_model.config.hidden_size

        # score function args
        self.score_function = score_function_args['score_function'] if 'score_function' in score_function_args else 'base'

        if self.task.lower() == 'cifar':
            self.input_model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            self.input_model.maxpool = nn.Identity()
        
        self.input_model.fc = nn.Linear(
            self.input_model.fc.in_features, self.label_model_hidden, bias=False
        )
        
        if self.score_function == 'mlp':
            self.linear_hidden_1 = score_function_args['mlp_hidden_1']
            self.linear_hidden_2 = score_function_args['mlp_hidden_2']
            self.linear_dropout = score_function_args['mlp_dropout_rate']

            self.score_function_layer = nn.Sequential(
                                            nn.Linear(2 * self.label_model_hidden, self.linear_hidden_1),
                                            nn.ReLU(),
                                            nn.Dropout(self.linear_dropout),
                                            nn.ReLU(),
                                            nn.Linear(self.linear_hidden_1, self.linear_hidden_2),
                                            nn.ReLU(),
                                            nn.Dropout(self.linear_dropout),
                                            nn.Linear(self.linear_hidden_2, 1)
                                        )
            
    def forward(self, batch):
        # batch: (x, label)
        input_data, label_batch = batch
        
        input_rep = self.input_model(input_data) # (batch_size, hidden_size)

        label_batch = {k: v.squeeze(0) for k, v in label_batch.items()}
        with torch.set_grad_enabled(self.train_args['tune_label_model']):
            label_rep = self.label_model(**label_batch).pooler_output # (num_descrition * n_class, hidden_size)
            if self.num_description > 1:
                if self.multi_description_aggregation == 'concat':
                    label_rep = label_rep.reshape(-1, self.label_model_hidden) # (n_class, num_description * hidden_size) -> (n_class, label_model_hidden_size)
                if self.multi_description_aggregation == 'mean':
                    label_rep = label_rep.reshape(-1, self.num_description, self.label_model_hidden // self.num_description)
                    label_rep = torch.mean(label_rep, axis=(1)) # (n_class, hidden_size) -> (n_class, label_model_hidden_size)

            label_rep = label_rep.t() # (label_model_hidden_size, n_class)

        
        if self.score_function == 'base':
            logits = input_rep @ label_rep
        else:
            if self.score_function == 'mlp':
                # label_rep (batch_size, hidden_size)
                batch_size, hidden_size = input_rep.shape[0], input_rep.shape[1]
                n_class = label_rep.shape[1]

                input_rep_repeat = input_rep.unsqueeze(1).repeat(1, n_class, 1)
                label_rep_repeat = label_rep.t().unsqueeze(0).repeat(batch_size, 1, 1)

                combine_rep = torch.cat([input_rep_repeat, label_rep_repeat], dim=2)
                combine_rep = combine_rep.reshape(-1, 2 * hidden_size)
                
                logits = self.score_function_mlp(combine_rep).reshape((batch_size, -1))

        return logits