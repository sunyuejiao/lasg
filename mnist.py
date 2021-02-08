import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

""" set random seed """
seed = 2020
torch.manual_seed(seed)
np.random.seed(seed)

""" configuration """
num_workers = 20
num_sample = 60000
num_classes = 10
batch_size = 10000
devices = [torch.device('cuda', x%2+0) for x in range(num_workers)]
device_s = torch.device('cpu')
lr = 0.01
print_iter = 50
num_epochs = 20
path = './results_mnist_worker_20/'

""" load train and test datasets """
trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                      download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.MNIST(root='./data', train=False, 
                                     download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                         shuffle=False, num_workers=0)

from utils_gpu import *
alldata = sort_dataset(trainset, num_classes, num_sample)
#alldata = []
#for i in range(num_sample):
#    alldata.append(trainset[i])
#random.shuffle(alldata)

batch_ratio = 0.002
nsample = [ (60000 // num_workers) for i in range(num_workers)]
#nsample = [2000, 3000, 4000, 5000, 6000, 6000, 7000, 8000, 9000, 10000]
pointer = 0
subtrainloader = []
weights = []
for i in range(num_workers):
    subtrainloader.append(torch.utils.data.DataLoader(alldata[pointer:pointer+nsample[i]], batch_size=int(batch_ratio*nsample[i]),
                                          shuffle=True, num_workers=1)) #num_workers = num_workers
    pointer = pointer + nsample[i]
    weights.append(nsample[i]/num_sample)
alltrainloader = torch.utils.data.DataLoader(alldata, batch_size=batch_size, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)

""" define model """
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        #self.fc = nn.Linear(28*28*1, 10)
    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.elu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        #x = x.view(-1, 28*28*1)
        #x = self.fc(x)
        return x
    
""" define loss """
criterion = nn.CrossEntropyLoss()
