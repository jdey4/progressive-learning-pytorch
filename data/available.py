from torchvision import datasets, transforms
from data.manipulate import UnNormalize
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import librosa
import os

#jd's version to load spoken digit
def load_spoken_digit(path_recordings):
    file = os.listdir(path_recordings)

    audio_data = []  # audio data
    x_spec = []  # STFT spectrogram
    x_spec_mini = []  # resized image, 28*28
    y_number = []  # label of number
    y_speaker = []  # label of speaker

    for i in file:
        x, sr = librosa.load(path_recordings + i, sr=8000)
        x_stft = librosa.stft(x, n_fft=128)  # Extract STFT
        x_stft_db = librosa.amplitude_to_db(abs(x_stft))  # Convert an amplitude spectrogram to dB-scaled spectrogram
        x_stft_db_mini = cv2.resize(x_stft_db, (28, 28))  # Resize into 28 by 28
        y_n = i[0]  # number
        y_s = i[2]  # first letter of speaker's name

        x_spec.append(x_stft_db)
        x_spec_mini.append(x_stft_db_mini)
        y_number.append(y_n)
        y_speaker.append(y_s)

    x_spec_mini = np.array(x_spec_mini)
    y_number = np.array(y_number).astype(int)
    y_speaker = np.array(y_speaker)

    y_s = np.unique(y_speaker)
    y_number = np.array(y_number)
    for ii, speaker in enumerate(y_s):
        idx = np.where(y_speaker==speaker)[0]
        y_number[idx] += ii*10 

    return x_spec_mini, y_number


class SpokenDigit(Dataset):

    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform
        self.X, self.y = load_spoken_digit('/Users/jayantadey/progressive-learning-pytorch/data/free-spoken-digit-dataset/recordings/')
        self.X = self.X.reshape(3000,1,28,28)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        sample = (self.X[index], self.y[index])
        return sample

# Specify available data-sets
AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
    'cifar10': datasets.CIFAR10,
    'cifar100': datasets.CIFAR100,
    'spoken_digit': SpokenDigit,
}


# Specify available transforms
AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'mnist28': [
        transforms.ToTensor(),
    ],
    'cifar10': [
        transforms.ToTensor(),
    ],
    'cifar100': [
        transforms.ToTensor(),
    ],
    'spoken_digit': [
        transforms.ToTensor(),
    ],
    'cifar10_norm': [
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ],
    'cifar100_norm': [
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
    ],
    'cifar10_denorm': UnNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    'cifar100_denorm': UnNormalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
    'augment_from_tensor': [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ],
    'augment': [
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
    ],
}


# Specify configurations of available data-sets
DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist28': {'size': 28, 'channels': 1, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
    'spoken_digit': {'size': 28, 'channels': 1, 'classes': 60},
}