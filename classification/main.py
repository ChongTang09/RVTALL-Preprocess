import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import tqdm
import glob
from sklearn.utils import shuffle
from scipy.io import savemat
from PIL import Image

from train import Trainer
from models import MultiInResNet
from data_preprocess import load_data, preprocess

NUM_INPUTS = 1
SPLIT_RATE = 0.5

root_radar = r'D:\Glasgow\RVTALL\processed_cut_data\radar_processed'
root_uwb = r'D:\Glasgow\RVTALL\processed_cut_data\uwb_processed'
subjects = [str(i) for i in range(1, 11)]
sentences = ['sentences_'+str(i) for i in range(1, 11)]
words = ['word_'+str(i) for i in range(1, 16)]
vowels = ['vowel_'+str(i) for i in range(1, 6)]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Machine has {} GPUs".format(torch.cuda.device_count()))

# classifier
classifier = MultiInResNet(num_inputs=NUM_INPUTS,
                           num_classes=5, 
                           num_in_convs=[0], 
                           in_channels=[1], 
                           out1_channels=[0],
                           model='resnet18')

# trainloader and testloader
sent_samples = []
word_samples = []
vow_samples = []

sent_labels = []
word_labels = []
vow_labels = []

print('Data Loading...')
for sub in tqdm.tqdm(subjects):
    # for idx, sent in enumerate(sentences):
    #     sent_labels += [int(idx)]*len(glob.glob(root_uwb+'/'+sub+'/'+sent+'/*.png'))
    #     _sent_samples = load_data(root_uwb+'/'+sub+'/'+sent, 'uwb')
    for idx, word in enumerate(words):
        word_labels += [int(idx)]*len(glob.glob(root_uwb+'/'+sub+'/'+word+'/*.png'))
        _word_samples = load_data(root_uwb+'/'+sub+'/'+word, 'uwb')
    # for idx, vowel in enumerate(vowels):
    #     vow_labels += [int(idx)]*len(glob.glob(root_uwb+'/'+sub+'/'+vowel+'/*.png'))
    #     _vow_samples = load_data(root_uwb+'/'+sub+'/'+vowel, 'uwb')

        # sent_samples += _sent_samples
        word_samples += _word_samples
        # vow_samples += _vow_samples

data, label = word_samples, word_labels

for idx, sample in enumerate(data):
    data[idx] = preprocess(data[idx])

data, label = shuffle(data, label)

data = torch.stack(data, dim=0)
label = torch.Tensor(label).long()

# split train test data
train_num = int(data.size(0)*SPLIT_RATE)
train_X, train_Y = data[0:train_num], label[0:train_num]
test_X, test_Y = data[train_num:], label[train_num:]

trainset = TensorDataset(train_X, train_Y)
testset = TensorDataset(test_X, test_Y)
trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
testloader = DataLoader(testset, batch_size=8, shuffle=True)

# optimizers
lr = 1e-4
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

# Save model
torch.save(trainier.classifier.module.state_dict(), 'classifer.pt')