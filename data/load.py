import copy
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import ConcatDataset
from data.available import AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, DATASET_CONFIGS
from data.manipulate import ReducedDataset, ReducedSubDataset, SubDataset, TransformedDataset, GetSlotDataset, permutate_image_pixels, GetShuffledDataset
import os
from random import sample
import cv2 

#JD's change
TRAIN_DATADIR = '/Users/jayantadey/Downloads/LargeFineFoodAI/Train' #'/cis/home/jdey4/LargeFineFoodAI/Train'
VAL_DATADIR = '/Users/jayantadey/Downloads/LargeFineFoodAI/Val'

CATEGORIES = list(range(20))
SAMPLE_PER_CLASS = 60
NUM_CLASS_PER_TASK = 20
IMG_SIZE = 50

###################################################
class MyDataloader(torch.utils.data.Dataset):
	def __init__(self, X, Y):
		self.images = X / 255.
		self.labels = torch.from_numpy(Y)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		return torch.from_numpy(self.images[idx].transpose((2, 0, 1))).float(), self.labels[idx]


def get_food_dataset(tasks=50):
    train_datasets  = []
    test_datasets = []

    for task in range(tasks):
        train_X = []
        train_y = []
        test_X = []
        test_y = []
        
        categories_to_consider = range(task*NUM_CLASS_PER_TASK,(task+1)*NUM_CLASS_PER_TASK)
        for category in categories_to_consider:
            path = os.path.join(TRAIN_DATADIR, str(category))

            images = os.listdir(path)
            total_images = len(images)
            train_indx = sample(range(total_images), SAMPLE_PER_CLASS)
            test_indx = np.delete(range(total_images), train_indx)
            for ii in train_indx:
                image_data = cv2.imread(
                        os.path.join(path, images[ii])
                    )
                resized_image = cv2.resize(
                    image_data, 
                    (IMG_SIZE, IMG_SIZE)
                )
                train_X.append(
                    resized_image
                )
                train_y.append(
                    category
                )
            for ii in test_indx:
                image_data = cv2.imread(
                        os.path.join(path, images[ii])
                    )
                resized_image = cv2.resize(
                    image_data, 
                    (IMG_SIZE, IMG_SIZE)
                )
                test_X.append(
                    resized_image
                )
                test_y.append(
                    category
                )

        train_X = np.array(train_X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
        train_y = np.array(train_y)
        test_X = np.array(test_X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
        test_y = np.array(test_y)

        train_datasets.append(
            MyDataloader(
                train_X,
                train_y
            )
        )
        test_datasets.append(
            MyDataloader(
                test_X,
                test_y
            )
        )
    
    return train_datasets, test_datasets

    
def get_dataset(name, shift, type='train', download=True, capacity=None, permutation=None, dir='./store/datasets',
                verbose=False, augment=False, normalize=False, target_transform=None, valid_prop=0.):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'mnist' if name in ('mnist28') else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    transforms_list = [*AVAILABLE_TRANSFORMS['augment']] if augment else []
    transforms_list += [*AVAILABLE_TRANSFORMS[name]]
    if normalize:
        transforms_list += [*AVAILABLE_TRANSFORMS[name+"_norm"]]
    if permutation is not None:
        transforms_list.append(transforms.Lambda(lambda x, p=permutation: permutate_image_pixels(x, p)))
    dataset_transform = transforms.Compose(transforms_list)

    # load data-set
    dataset_train = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=True,
                            download=download, transform=dataset_transform, target_transform=target_transform)
    dataset_test = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    #JD's change
    dataset = ConcatDataset([dataset_train, dataset_test])
    dataset = GetSlotDataset(dataset, shift=shift, type=type)
    #dataset = GetShuffledDataset(dataset, shift=shift, type=type)

    #############

    # if relevant, select "train" or "validation"-set from training-part of data
    # NOTE: this split assumes order of items in training-dataset is random!
    # (e.g., not first all samples from clas 1, then all samples from class 2, etc.)
    if (type=='train' or type=='valid') and valid_prop>0:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(valid_prop * dataset_size))
        if type=='train':
            indices_to_use = indices[split:]
        elif type=='valid':
            indices_to_use = indices[:split]
        dataset = ReducedDataset(dataset, indices_to_use)

    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset_copy = copy.deepcopy(dataset)
        dataset = ConcatDataset([dataset_copy for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset


##-------------------------------------------------------------------------------------------------------------------##


def get_singletask_experiment(name, data_dir="./store/datasets", normalize=False, augment=False, verbose=False):
    '''Load, organize and return train- and test-dataset for requested single-task experiment.'''

    # Define data-type
    if name == "MNIST":
        data_type = 'mnist'
    elif name == "MNIST28":
        data_type = 'mnist28'
    elif name == "CIFAR10":
        data_type = 'cifar10'
    elif name == "CIFAR100":
        data_type = 'cifar100'
    else:
        raise ValueError('Given undefined experiment: {}'.format(name))

    # Get config-dict and data-sets
    config = DATASET_CONFIGS[data_type]
    config['normalize'] = normalize
    if normalize:
        config['denormalize'] = AVAILABLE_TRANSFORMS[data_type+"_denorm"]
    trainset = get_dataset(data_type, type='train', dir=data_dir, verbose=verbose, normalize=normalize, augment=augment)
    testset = get_dataset(data_type, type='test', dir=data_dir, verbose=verbose, normalize=normalize)

    # Return tuple of data-sets and config-dictionary
    return (trainset, testset), config


def get_multitask_experiment(name, tasks, data_dir="./store/datasets", normalize=False, augment=False,
                             only_config=False, verbose=False, exception=False, only_test=False, max_samples=None):
    '''Load, organize and return train- and test-dataset for requested multi-task experiment.'''

    ## NOTE: option 'normalize' and 'augment' only implemented for CIFAR-based experiments.

    # depending on experiment, get and organize the datasets
    if name == 'permMNIST':
        # configurations
        config = DATASET_CONFIGS['mnist']
        classes_per_task = 10
        if not only_config:
            # prepare dataset
            if not only_test:
                train_dataset = get_dataset('mnist', type="train", permutation=None, dir=data_dir,
                                            target_transform=None, verbose=verbose)
            test_dataset = get_dataset('mnist', type="test", permutation=None, dir=data_dir,
                                       target_transform=None, verbose=verbose)
            # generate permutations
            if exception:
                permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(tasks-1)]
            else:
                permutations = [np.random.permutation(config['size']**2) for _ in range(tasks)]
            # specify transformed datasets per task
            train_datasets = []
            test_datasets = []
            for task_id, perm in enumerate(permutations):
                target_transform = transforms.Lambda(lambda y, x=task_id: y + x*classes_per_task)
                if not only_test:
                    train_datasets.append(TransformedDataset(
                        train_dataset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                        target_transform=target_transform
                    ))
                test_datasets.append(TransformedDataset(
                    test_dataset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                    target_transform=target_transform
                ))
    elif name == 'splitMNIST':
        # check for number of tasks
        if tasks>10:
            raise ValueError("Experiment '{}' cannot have more than 10 tasks!".format(name))
        # configurations
        config = DATASET_CONFIGS['mnist28']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            # prepare train and test datasets with all classes
            if not only_test:
                mnist_train = get_dataset('mnist28', type="train", dir=data_dir, target_transform=target_transform,
                                          verbose=verbose)
            mnist_test = get_dataset('mnist28', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = None
                if not only_test:
                    train_datasets.append(SubDataset(mnist_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(mnist_test, labels, target_transform=target_transform))
    elif name == 'CIFAR100':
        # check for number of tasks
        if tasks>100:
            raise ValueError("Experiment 'CIFAR100' cannot have more than 100 tasks!")
        # configurations
        config = DATASET_CONFIGS['cifar100']
        classes_per_task = int(np.floor(100 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = list(range(100)) #np.random.permutation(list(range(100)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            if not only_test:
                cifar100_train = get_dataset('cifar100', shift=shift, type="train", dir=data_dir, normalize=normalize,
                                             augment=augment, target_transform=target_transform, verbose=verbose)
            cifar100_test = get_dataset('cifar100', shift=shift, type="test", dir=data_dir, normalize=normalize,
                                        target_transform=target_transform, verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = None
                if not only_test:
                    if max_samples is None:
                        train_datasets.append(SubDataset(cifar100_train, labels, target_transform=target_transform))
                    else:
                        train_datasets.append(ReducedSubDataset(cifar100_train, labels,
                                                                target_transform=target_transform, max=max_samples))
                test_datasets.append(SubDataset(cifar100_test, labels, target_transform=target_transform))
    elif name == 'food1k':
        config = DATASET_CONFIGS['food1k']
        classes_per_task = int(np.floor(1000 / tasks))

        train_datasets, test_datasets = get_food_dataset(tasks)
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task*tasks
    config['normalize'] = normalize if name=='CIFAR100' else False
    if config['normalize']:
        config['denormalize'] = AVAILABLE_TRANSFORMS["cifar100_denorm"]

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)