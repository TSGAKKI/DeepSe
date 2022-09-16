import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import os
import numpy as np

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

BATCH_SIZE = 1000

def get_EEGDenoiseNet():

    raw_eeg = np.load('/data/Liulei/preprocess/DeepSeparator-main/data/train_input.npy')
    clean_eeg = np.load('/data/Liulei/preprocess/DeepSeparator-main/data/train_output.npy')

    artifact1 = np.load('/data/Liulei/preprocess/DeepSeparator-main/data/EOG_all_epochs.npy')
    artifact2 = np.load('/data/Liulei/preprocess/DeepSeparator-main/data/EMG_all_epochs.npy')

    test_input = np.load('/data/Liulei/preprocess/DeepSeparator-main/data/test_input.npy')
    test_output = np.load('/data/Liulei/preprocess/DeepSeparator-main/data/test_output.npy')

    artifact1 = standardization(artifact1)
    artifact2 = standardization(artifact2)
    artifact = np.concatenate((artifact1, artifact2), axis=0)

    indicator1 = np.zeros(raw_eeg.shape[0])
    indicator2 = np.ones(artifact.shape[0])
    indicator3 = np.zeros(clean_eeg.shape[0])
    indicator = np.concatenate((indicator1, indicator2, indicator3), axis=0)

    train_input = np.concatenate((raw_eeg, artifact, clean_eeg), axis=0)
    train_output = np.concatenate((clean_eeg, artifact, clean_eeg), axis=0)

    indicator = torch.from_numpy(indicator)
    indicator = indicator.unsqueeze(1)

    train_input = torch.from_numpy(train_input)
    train_output = torch.from_numpy(train_output)

    train_torch_dataset = Data.TensorDataset(train_input, indicator, train_output)

    train_loader = Data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_input = torch.from_numpy(test_input)
    test_output = torch.from_numpy(test_output)

    test_indicator = np.zeros(test_input.shape[0])
    test_indicator = torch.from_numpy(test_indicator)
    test_indicator = test_indicator.unsqueeze(1)

    test_torch_dataset = Data.TensorDataset(test_input, test_indicator, test_output)

    test_loader = Data.DataLoader(
        dataset=test_torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader,test_loader