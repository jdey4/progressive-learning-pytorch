import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

#jd's version to manipulate the data
class GetSlotDataset(Dataset):

    def __init__(self, datatset_to_process, shift, type='train'):
        super().__init__()
        self.datatset = datatset_to_process
        self.indeces = []

        label = np.asarray([lbl for _,lbl in self.datatset])
        idx = np.asarray([np.where(label==i) for i in np.unique(label)])
        
        if type == 'train':
            for ii in range(len(idx)):
                self.indeces.extend(
                    list(
                        np.roll(idx[ii],(shift-1)*100)[0][0:500]
                    )
                )
        else:
            for ii in range(len(idx)):
                self.indeces.extend(
                    list(
                        np.roll(idx[ii],(shift-1)*100)[0][500:600]
                    )
                )
                
    def __len__(self):
        return len(self.indeces)

    def __getitem__(self, index):
        return self.datatset[self.indeces[index]]


class GetSpokenDigitDataset(Dataset):

    def __init__(self, datatset_to_process, seed, type='train'):
        super().__init__()
        self.datatset = datatset_to_process
        X = np.asarray([x for x,_ in self.datatset])
        label = np.asarray([lbl for _,lbl in self.datatset])
        self.indeces = []

        np.random.seed(seed)
        train_x, test_x, train_y, test_y = train_test_split(
                                                X, 
                                                label, 
                                                test_size=0.25
                                                )
        if type == 'train':
            self.sample = train_x
            self.indeces = train_y
        else:
            self.sample = test_x
            self.indeces = test_y
        
    def __len__(self):
        return len(self.indeces)

    def __getitem__(self, index):
        return (self.sample[self.indeces[index]], self.indeces[index])

#jd's version to randomly shuffle class labels for tasks 2-10
class GetShuffledDataset(Dataset):
    
    def __init__(self, datatset_to_process, slot, shift, type='train'):
        super().__init__()
        self.datatset = datatset_to_process
        self.indeces = []

        label = np.asarray([lbl for _,lbl in self.datatset])
        idx = [np.where(label==i) for i in np.unique(label)]

        shuffled_label = []
        for task in range(10):
            _tmp = [] 
            for cls in range(10*task,10*(task+1)):
                _tmp.extend(label[idx[cls]])
            
            if task > 0:
                np.random.shuffle(_tmp)
            
            shuffled_label.extend(_tmp)

        shuffled_label = np.asarray(shuffled_label)
        
        label = shuffled_label 

        idx = np.asarray([np.where(label==i) for i in np.unique(label)])
        
        if type == 'train':
            for ii in range(len(idx)):
                self.indeces.extend(
                    list(
                        np.roll(idx[ii],(shift-1)*100)[0][(slot-1)*50:slot*50]
                    )
                )
        else:
            for ii in range(len(idx)):
                self.indeces.extend(
                    list(
                        np.roll(idx[ii],(shift-1)*100)[0][500:600]
                    )
                )
                
    def __len__(self):
        return len(self.indeces)

    def __getitem__(self, index):
        return self.datatset[self.indeces[index]]
 

class ReducedDataset(Dataset):
    '''To reduce a dataset, taking only samples corresponding to provided indeces.
    This is useful for splitting a dataset into a training and validation set.'''

    def __init__(self, original_dataset, indeces):
        super().__init__()
        self.dataset = original_dataset
        self.indeces = indeces

    def __len__(self):
        return len(self.indeces)

    def __getitem__(self, index):
        return self.dataset[self.indeces[index]]


class ReducedSubDataset(Dataset):
    '''To reduce & sub-sample a dataset, taking only those samples with label in [sub_labels] and at most [max] of them.

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None, max=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        counts = {}
        for label in sub_labels:
            counts[label] = 0
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                if counts[label]<max:
                    counts[label] += 1
                    self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


class ExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, exemplar_sets, target_transform=None):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
        return (image, class_id_to_return)


class TransformedDataset(Dataset):
    '''To modify an existing dataset with a transform.
    This is useful for creating different permutations of MNIST without loading the data multiple times.'''

    def __init__(self, original_dataset, transform=None, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (input, target) = self.dataset[index]
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return (input, target)


#----------------------------------------------------------------------------------------------------------#


def permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].

    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''

    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image


#----------------------------------------------------------------------------------------------------------#


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """Denormalize image, either single image (C,H,W) or image batch (N,C,H,W)"""
        batch = (len(tensor.size())==4)
        for t, m, s in zip(tensor.permute(1,0,2,3) if batch else tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor