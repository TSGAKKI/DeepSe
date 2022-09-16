import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import os
import numpy as np
from network import DeepSeparator
from get_isruc import *
from data_prepare import *
import utils 

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


BATCH_SIZE = 1000
learning_rate = 1e-4
epochs = 50

mini_loss = 1

print_loss_frequency = 1
print_train_accuracy_frequency = 1
test_frequency = 1

model_name = 'DeepSeparator'
model = DeepSeparator()
loss = nn.MSELoss(reduction='mean')

log = utils.get_logger('/data/Liulei/preprocess/DeepSeparator-main/code/checkpiont', 'train')

#--EEGDenoiseNet

raw_eeg = np.load('../data/train_input.npy')
clean_eeg = np.load('../data/train_output.npy')

artifact1 = np.load('../data/EOG_all_epochs.npy')
artifact2 = np.load('../data/EMG_all_epochs.npy')

test_input = np.load('../data/test_input.npy')
test_output = np.load('../data/test_output.npy')

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
log.info('Building dataset...')

#ISRUC
dataset='ISRUC'
noise_type='full'
batch_size=1000
if dataset == 'ISRUC':
    fold_clean, fold_noise, fold_contaminated, fold_len = get_isruc('/data/Liulei/preprocess/data','full')

DataGenerator = kFoldGenerator(fold_clean, fold_noise, fold_contaminated, batch_size , noise_type)

print("torch.cuda.is_available() = ", torch.cuda.is_available())

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if os.path.exists('/data/Liulei/preprocess/DeepSeparator-main/code/checkpiont' + model_name + '.pkl'):
    print('load model')
    log.info('load model')
    model.load_state_dict(torch.load('/data/Liulei/preprocess/DeepSeparator-main/code/checkpiont' + model_name + '.pkl'))
histories = []


for i in range(1):
    # Train
    train_loader, test_loader = DataGenerator.getFold(i)
    print(128*'-')
    log.info('fold {} is running'.format(i+1))
    for epoch in range(epochs):

        train_acc = 0
        train_loss = 0

        total_train_loss_per_epoch = 0
        average_train_loss_per_epoch = 0
        train_step_num = 0

        for (train_input, indicator, train_output) in train_loader:

            train_step_num += 1

            indicator = indicator.float().to(device)
            train_input = train_input.float().to(device)
            train_output = train_output.float().to(device)

            optimizer.zero_grad()

            train_preds = model(train_input, indicator)

            train_loss = loss(train_preds, train_output)

            total_train_loss_per_epoch += train_loss.item()

            train_loss.backward()
            optimizer.step()

        average_train_loss_per_epoch = total_train_loss_per_epoch / train_step_num

        if epoch % print_loss_frequency == 0:
            print('train loss: {}'.format(average_train_loss_per_epoch))
            log.info('train loss: {}'.format(average_train_loss_per_epoch))

        test_step_num = 0
        total_test_loss_per_epoch = 0
        average_test_loss_per_epoch = 0

        if epoch % test_frequency == 0:
            with torch.no_grad():
                for step, (test_input, test_indicator, test_output) in enumerate(test_loader):

                    test_step_num += 1

                    test_indicator = test_indicator.float().to(device)

                    test_input = test_input.float().to(device)
                    test_output = test_output.float().to(device)

                    test_preds = model(test_input, 0)

                    test_loss = loss(test_preds, test_output)

                    total_test_loss_per_epoch += test_loss.item()

            average_test_loss_per_epoch = total_test_loss_per_epoch / test_step_num

            print('--------------test loss: {}'.format(average_test_loss_per_epoch))
            log.info('--------------test loss: {}'.format(average_test_loss_per_epoch))

        # if average_test_loss_per_epoch < mini_loss:
        #     print('save model')
        #     torch.save(model.state_dict(), '/data/Liulei/preprocess/DeepSeparator-main/code/checkpiont/' + model_name + '.pkl')
        #     mini_loss = average_test_loss_per_epoch
