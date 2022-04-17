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

CifarSuperClass2Classes = {
        "aquatic_mammals": ["otter", "beaver", "whale", "dolphin", "seal"],
        "fish": ["trout", "aquarium_fish", "shark", "flatfish", "ray"],
        "flowers": ["poppy", "rose", "orchid", "sunflower", "tulip"],
        "food_containers": ["plate", "bowl", "bottle", "can", "cup"],
        "fruit_and_vegetables": [
            "orange",
            "apple",
            "pear",
            "sweet_pepper",
            "mushroom",
        ],
        "household_electrical_devices": [
            "clock",
            "keyboard",
            "telephone",
            "television",
            "lamp",
        ],
        "household_furniture": ["table", "chair", "couch", "wardrobe", "bed"],
        "insects": ["caterpillar", "bee", "cockroach", "beetle", "butterfly"],
        "large_carnivores": ["leopard", "lion", "tiger", "bear", "wolf"],
        "large_man-made_outdoor_things": [
            "house",
            "bridge",
            "skyscraper",
            "road",
            "castle",
        ],
        "large_natural_outdoor_scenes": [
            "forest",
            "cloud",
            "plain",
            "mountain",
            "sea",
        ],
        "large_omnivores_and_herbivores": [
            "kangaroo",
            "cattle",
            "elephant",
            "camel",
            "chimpanzee",
        ],
        "medium_mammals": ["raccoon", "fox", "porcupine", "possum", "skunk"],
        "non-insect_invertebrates": ["snail", "lobster", "spider", "worm", "crab"],
        "people": ["girl", "woman", "man", "baby", "boy"],
        "reptiles": ["turtle", "snake", "lizard", "crocodile", "dinosaur"],
        "small_mammals": ["mouse", "shrew", "hamster", "squirrel", "rabbit"],
        "trees": [
            "palm_tree",
            "willow_tree",
            "pine_tree",
            "oak_tree",
            "maple_tree",
        ],
        "vehicles_1": ["bus", "bicycle", "motorcycle", "train", "pickup_truck"],
        "vehicles_2": ["streetcar", "tank", "lawn_mower", "tractor", "rocket"],
    } 

CifarSuperClassConfig = {
    'superclasses': (
        "aquatic_mammals",
        "fish",
        "flowers",
        "food_containers",
        "fruit_and_vegetables",
        "household_electrical_devices",
        "household_furniture",
        "insects",
        "large_carnivores",
        "large_man-made_outdoor_things",
        "large_natural_outdoor_scenes",
        "large_omnivores_and_herbivores",
        "medium_mammals",
        "non-insect_invertebrates",
        "people",
        "reptiles",
        "small_mammals",
        "trees",
        "vehicles_1",
        "vehicles_2",
    ),
    'val_superclasses': (
        "large_omnivores_and_herbivores",
        "people",
        "medium_mammals",
        "large_man-made_outdoor_things",
        "insects",
        "household_electrical_devices",
        "food_containers",
        "fish",
        "flowers",
        "vehicles_2",
    ),
    'test_superclasses': (
        "small_mammals",
        "reptiles",
        "non-insect_invertebrates",
        "large_natural_outdoor_scenes",
        "large_carnivores",
        "household_furniture",
        "fruit_and_vegetables",
        "aquatic_mammals",
        "trees",
        "vehicles_1",
    )
}

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
        # input dataloader
        # transform applied later
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.cache_dir, train=True, download=False
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=self.cache_dir, train=True, download=False
        )

        if self.general_args['run_test']:
            val_dataset = torchvision.datasets.CIFAR100(
                root=self.cache_dir, train=False, download=False
            )
        
        train_idx, val_idx = train_test_split(
            list(range(len(train_dataset))),
            test_size=self.general_args['val_size'],
            random_state=self.general_args['split_seed'],
            shuffle=True,
            stratify=train_dataset.targets
        )

        # furthur filter val dataset to remove instances whose superclass is not in the val superclass
        if self.val_level == 'coarse':
            val_final_idx = []
            val_class_ids = [
                self.all_class_label.str2int(x) for x in self.classes \
                    if self.classes_to_superclass[x] in self.val_superclasses
            ]
            for idx in val_idx:
                _, class_id = val_dataset[idx]
                if class_id in val_class_ids:
                    val_final_idx.append(idx)
            val_idx = val_final_idx

        train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(val_dataset, val_idx)

        train_target_transform = lambda x: x if self.train_level == 'fine' \
            else lambda x: self.train_class_label.str2int(
                CifarSuperClass2Classes[self.all_class_label.int2str(x)])
        
        val_target_transform = lambda x: x if self.val_level == 'fine'\
            else lambda x: self.val_class_label.str2int(
                CifarSuperClass2Classes[self.all_class_label.int2str(x)]
            )
        
        train_dataset.dataset.transforms = StandardTransform(
            transform=self.train_transform, target_transform=train_target_transform
        )
        val_dataset.dataset.transforms = StandardTransform(
            transform=self.eval_transform, target_transform=val_target_transform
        )

        self.input_dataset['train'] = train_dataset
        self.input_dataset['val'] = val_dataset
        