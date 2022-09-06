#%%
import copy
import torchvision
import torch
import tarfile
import os
import cv2
import imageio 
import numpy as np
from torchvision import transforms
from torch.utils.data import ConcatDataset
from data.available import AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, DATASET_CONFIGS
from data.manipulate import ReducedDataset, ReducedSubDataset, SubDataset, TransformedDataset, GetSlotDataset, permutate_image_pixels, GetShuffledDataset
import pickle
#%%
#JD's change
class MyDataloader(torch.utils.data.Dataset):
	def __init__(self, X, Y):
		self.images = X / 255.
		self.labels = torch.from_numpy(Y)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		return torch.from_numpy(self.images[idx].transpose((2, 0, 1))).float(), self.labels[idx]


def get_nomnist(task_id):
	"""
    Parses and returns the downloaded notMNIST dataset
    """
	classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
	tar_path = "./data/notMNIST_small.tar"
	tmp_path = "./data/tmp"

	img_arr = []
	lab_arr = []

	with tarfile.open(tar_path) as tar:
		tar_root = tar.next().name
		for ind, c in enumerate(classes):
			files = [f for f in tar.getmembers() if f.name.startswith(tar_root + '/' + c)]
			if not os.path.exists(tmp_path):
				os.mkdir(tmp_path)
			for f in files:
				f_obj = tar.extractfile(f)
				try:
					arr = np.asarray(imageio.imread(f_obj))
					img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
					img = cv2.resize(img, (32, 32))
					img_arr.append(np.asarray(img))
					lab_arr.append(ind + task_id * len(classes))
				except:
					continue
	os.rmdir(tmp_path)
	return np.array(img_arr), np.array(lab_arr)

def get_5_datasets(task_id, DATA, get_val=False):
	"""
    Returns the data loaders for a single task of 5-dataset
    :param task_id: Current task id
    :param DATA: Dataset class from torchvision
    :param batch_size: Batch size
    :param get_val: Get validation set for grid search
    :return: Train, test and validation data loaders
    """
	if task_id in [0, 2]:
		transforms = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),

		])
	else:
		transforms = torchvision.transforms.Compose([
			torchvision.transforms.Resize(32),
			torchvision.transforms.Lambda(lambda x: x.convert('RGB')),
			torchvision.transforms.ToTensor(),
		])
	target_transform = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda y: y + task_id * 10)])

	# All datasets except notMNIST (task_id=3) are available in torchvision
	if task_id != 3:
		try:
			train_data = DATA('./data/', train=True, download=True, transform=transforms,
			                  target_transform=target_transform)
			test_data = DATA('./data/', train=False, download=True, transform=transforms,
			                 target_transform=target_transform)
		except:
			# Slighly different way to import SVHN
			train_data = DATA('./data/SVHN/', split='train', download=True, transform=transforms,
			                  target_transform=target_transform)
			test_data = DATA('./data/SVHN/', split='test', download=True, transform=transforms,
			                 target_transform=target_transform)
		#test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4,
		#                                          pin_memory=True)
	else:
		all_images, all_labels = get_nomnist(task_id)
		dataset_size = len(all_images)
		indices = list(range(dataset_size))
		split = int(np.floor(0.1 * dataset_size))
		np.random.shuffle(indices)
		train_indices, test_indices = indices[split:], indices[:split]
		train_data = MyDataloader(all_images[train_indices], all_labels[train_indices])
		test_data = MyDataloader(all_images[test_indices], all_labels[test_indices])
		
	return train_data, test_data

def get_5_datasets_tasks(num_tasks, get_val=False):
	"""
    Returns data loaders for all tasks of 5-dataset.
    :param num_tasks: Total number of tasks
    :param batch_size: Batch-size for training data
    :param get_val: Get validation set for grid search
    """
	datasets = {}
	data_list = [torchvision.datasets.CIFAR10,
	             torchvision.datasets.MNIST,
	             torchvision.datasets.SVHN,
	             'notMNIST',
	             torchvision.datasets.FashionMNIST]
	for task_id, DATA in enumerate(data_list):
		print('Loading Task/Dataset:', task_id)
		train_loader, test_loader = get_5_datasets(task_id, DATA)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets

class XYDataset(torch.utils.data.Dataset):
	"""
    Image pre-processing
    """

	def __init__(self, x, y, **kwargs):
		self.x, self.y = x, y

		# this was to store the inverse permutation in permuted_mnist
		# so that we could 'unscramble' samples and plot them
		for name, value in kwargs.items():
			setattr(self, name, value)

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		x, y = self.x[idx], self.y[idx]

		if type(x) != torch.Tensor:
			# mini_imagenet
			# we assume it's a path --> load from file
			x = self.transform(x)
			y = torch.Tensor(1).fill_(y).long().squeeze()
		else:
			x = x.float() / 255.
			y = y.long()

		# for some reason mnist does better \in [0,1] than [-1, 1]
		if self.source == 'mnist':
			return x, y
		else:
			return (x - 0.5) * 2, y


def get_miniimagenet(get_val=False):
	"""
    Import mini-imagenet dataset and split it into multiple tasks with disjoint set of classes
    Implementation is based on the one provided by:
        Aljundi, Rahaf, et al. "Online continual learning with maximally interfered retrieval."
        arXiv preprint arXiv:1908.04742 (2019).
    :param args: Arguments for model/data configuration
    :param get_val: Get validation set for grid search
    :return: Train, test and validation data loaders
    """
	#args.use_conv = True
	#args.n_classes = 100
	# if args.multi == 1:
	#     args.tasks = 1
	#args.n_classes_per_task = args.n_classes
	# else:
	#     args.tasks = 20
	#args.n_classes_per_task = 5
	#args.input_size = (3, 84, 84)

	transform = transforms.Compose([
		# transforms.Resize(84),
		# transforms.CenterCrop(84),
		transforms.ToTensor(),
	])
	for i in ['train', 'test', 'val']:
		file = open("/Users/jayantadey/TAG/data/mini_imagenet/mini-imagenet-cache-" + i + ".pkl", "rb")
		file_data = pickle.load(file)
		data = file_data["image_data"]
		if i == 'train':
			main_data = data.reshape([64, 600, 84, 84, 3])
		else:
			app_data = data.reshape([(20 if i == 'test' else 16), 600, 84, 84, 3])
			main_data = np.append(main_data, app_data, axis=0)
	all_data = main_data.reshape((60000, 84, 84, 3))
	all_label = np.array([[i] * 600 for i in range(100)]).flatten()

	train_ds, test_ds = [], []
	current_train, current_test = None, None

	cat = lambda x, y: np.concatenate((x, y), axis=0)

	for i in range(100):
		class_indices = np.argwhere(all_label == i).reshape(-1)
		class_data = all_data[class_indices]
		class_label = all_label[class_indices]
		split = int(0.8 * class_data.shape[0])

		data_train, data_test = class_data[:split], class_data[split:]
		label_train, label_test = class_label[:split], class_label[split:]

		if current_train is None:
			current_train, current_test = (data_train, label_train), (data_test, label_test)
		else:
			current_train = cat(current_train[0], data_train), cat(current_train[1], label_train)
			current_test = cat(current_test[0], data_test), cat(current_test[1], label_test)

		if i % 5 == (5 - 1):
			train_ds += [current_train]
			test_ds += [current_test]
			current_train, current_test = None, None

	# build masks
	masks = []
	task_ids = [None for _ in range(20)]
	for task, task_data in enumerate(train_ds):
		labels = np.unique(task_data[1])  # task_data[1].unique().long()
		assert labels.shape[0] == 5
		mask = torch.zeros(100)
		mask[labels] = 1
		masks += [mask]
		task_ids[task] = labels

	task_ids = torch.from_numpy(np.stack(task_ids)).long()

	test_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'mini_imagenet', 'mask': y, 'task_ids': task_ids,
	                                                    'transform': transform}), test_ds, masks)
	if get_val:
		train_ds, val_ds = make_valid_from_train(train_ds)
		val_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'mini_imagenet', 'mask': y, 'task_ids': task_ids,
		                                                   'transform': transform}), val_ds, masks)
	else:
		val_ds = test_ds
	train_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'mini_imagenet', 'mask': y, 'task_ids': task_ids,
	                                                     'transform': transform}), train_ds, masks)

	return train_ds, test_ds, val_ds


def make_valid_from_train(dataset, cut=0.9):
	"""
    Split training data to get validation set
    :param dataset: Training dataset
    :param cut: Percentage of dataset to be kept for training purpose
    """
	tr_ds, val_ds = [], []
	for task_ds in dataset:
		x_t, y_t = task_ds

		# shuffle before splitting
		perm = torch.randperm(len(x_t))
		x_t, y_t = x_t[perm], y_t[perm]

		split = int(len(x_t) * cut)
		x_tr, y_tr = x_t[:split], y_t[:split]
		x_val, y_val = x_t[split:], y_t[split:]

		tr_ds += [(x_tr, y_tr)]
		val_ds += [(x_val, y_val)]

	return tr_ds, val_ds



def get_dataset(name, shift, slot, type='train', download=True, capacity=None, permutation=None, dir='./store/datasets',
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
    #dataset = GetSlotDataset(dataset, slot=slot, shift=shift, type=type)
    dataset = GetShuffledDataset(dataset, slot=slot, shift=shift, type=type)

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
                cifar100_train = get_dataset('cifar100', type="train", dir=data_dir, normalize=normalize,
                                             augment=augment, target_transform=target_transform, verbose=verbose)
            cifar100_test = get_dataset('cifar100', type="test", dir=data_dir, normalize=normalize,
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
    elif name == '5data':
        config = DATASET_CONFIGS['dataset5']
        classes_per_task = int(np.floor(50 / tasks))

        taskset = get_5_datasets_tasks(tasks)
        train_datasets, test_datasets = [taskset[i]['train'] for i in taskset], [taskset[i]['test'] for i in taskset]
    elif name == 'imagenet':
        config = DATASET_CONFIGS['imagenet']
        classes_per_task = int(np.floor(100 / tasks))

        train_dataset, test_dataset, _ = get_miniimagenet(tasks)
        #print(list(list(train_datasets)[0]), 'hi')
        train_datasets, test_datasets = [list(i) for i in list(train_dataset)], [list(i) for i in test_dataset]

    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task*tasks
    config['normalize'] = normalize if name=='CIFAR100' else False
    if config['normalize']:
        config['denormalize'] = AVAILABLE_TRANSFORMS["cifar100_denorm"]

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)