import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import tqdm
import glob
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
from scipy.io import savemat
from PIL import Image

from train import Trainer
from models import MultiInResNet
from data_preprocess import load_data, preprocess

NUM_INPUTS = 2
SPLIT_RATE = 0.8

root_fmcw = r'D:\Glasgow\RVTALL\processed_cut_data\radar_processed'
root_uwb = r'D:\Glasgow\RVTALL\processed_cut_data\uwb_processed'
subjects = [str(i) for i in range(1, 21)]
sentences = ['sentences_'+str(i) for i in range(1, 11)]
words = ['word_'+str(i) for i in range(1, 16)]
vowels = ['vowel_'+str(i) for i in range(1, 6)]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Machine has {} GPUs".format(torch.cuda.device_count()))

# classifier
classifier = MultiInResNet(num_inputs=NUM_INPUTS,
                           num_classes=15, 
                           num_in_convs=[1, 1], 
                           in_channels=[3, 3], 
                           out1_channels=[3, 3],
                           model='resnet18')

# trainloader and testloader
# sent_samples = []
# sent_labels = []

# vow_samples = []
# vow_labels = []
uwb_word_samples, fmcw_word_samples = [], []
word_labels = []

print('Data Loading...')
for sub in tqdm.tqdm(subjects):
    # for idx, sent in enumerate(sentences):
    #     sent_labels += [int(idx)]*len(glob.glob(root_uwb+'/'+sub+'/'+sent+'/*.png'))
    #     _sent_samples = load_data(root_uwb+'/'+sub+'/'+sent, 'uwb')
    for idx, word in enumerate(words):
        word_labels += [int(idx)]*len(glob.glob(root_uwb+'/'+sub+'/'+word+'/*.png'))
        _uwb_word_samples, _uwb_samples_idx = load_data(root_uwb+'/'+sub+'/'+word, 'uwb')
        _fmcw_word_samples, _fmcw_samples_idx = load_data(root_fmcw+'/'+sub+'/'+word.replace('_', ''), 'fmcw')

        common_elements = set(_uwb_samples_idx)&set(_fmcw_samples_idx)
    # for idx, vowel in enumerate(vowels):
    #     vow_labels += [int(idx)]*len(glob.glob(root_uwb+'/'+sub+'/'+vowel+'/*.png'))
    #     _vow_samples = load_data(root_uwb+'/'+sub+'/'+vowel, 'uwb')

        # sent_samples += _sent_samples
        uwb_word_samples += [_uwb_word_samples[_uwb_samples_idx.index(ele)] for ele in common_elements]
        fmcw_word_samples += [_fmcw_word_samples[_fmcw_samples_idx.index(ele)] for ele in common_elements]
        # vow_samples += _vow_samples

data_uwb, data_fmcw, label = uwb_word_samples, fmcw_word_samples, word_labels

for idx in range(len(data_uwb)):
    data_uwb[idx] = preprocess(data_uwb[idx])
    data_fmcw[idx] = preprocess(data_fmcw[idx])

temp = list(zip(data_uwb, data_fmcw, label))
random.shuffle(temp)
data_uwb, data_fmcw, label = zip(*temp)

data_uwb = torch.stack(data_uwb, dim=0)
data_fmcw = torch.stack(data_fmcw, dim=0)
label = torch.Tensor(label).long()

# split train test data
train_num = int(label.size(0)*SPLIT_RATE)
train_X_uwb, train_X_fmcw, train_Y = data_uwb[0:train_num], data_fmcw[0:train_num], label[0:train_num]
test_X_uwb, test_X_fmcw, test_Y = data_uwb[train_num:], data_fmcw[train_num:], label[train_num:]

trainset = TensorDataset(train_X_uwb, train_X_fmcw, train_Y)
testset = TensorDataset(test_X_uwb, test_X_fmcw, test_Y)
trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
testloader = DataLoader(testset, batch_size=8, shuffle=True)

# optimizers
lr = 1e-5
betas = (.5, .99)
optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=betas)
criterion = nn.CrossEntropyLoss()

# train model
epochs = 10

trainier = Trainer(num_inputs=NUM_INPUTS,
                    classifier=classifier,
                    optimizer=optimizer,
                    criterion=criterion,
                    print_every=1,
                    device=device,
                    use_cuda=use_cuda,
                    use_scheduler=False)

trainier.train(trainloader=trainloader, testloader=testloader, epochs=epochs)

cm = confusion_matrix([x.item() for x in trainier.ground_truth], [x.item() for x in trainier.predictions])
plt.figure()
sns.heatmap(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], annot=True, 
            fmt='.2%', cmap='Blues')
plt.show()
np.save('./classification/confusion_matrix/uwb_fmcw_word_cm.npy', cm)
# Save model
# torch.save(trainier.classifier.module.state_dict(), 'classifer.pt')