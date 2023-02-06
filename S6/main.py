from __future__ import print_function

"""### Loading required libraries"""

# load required libraries
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

#!pip install torchsummary
from torchsummary import summary

from tqdm import tqdm 
import numpy as np

#%matplotlib inline
import matplotlib.pyplot as plt

#!pip install albumentations==1.3.0
import albumentations as A
from albumentations.pytorch import ToTensorV2



import os
os.chdir('/content/drive/MyDrive/EVA8/S6/')

# loading model & utility functions
from model.model import Net
from utils.utils import imshow, train, test, evaluate



"""### Calculate mean and std deviation for imputation in course dropout"""

BATCH_SIZE = 16

test_transforms = transforms.Compose([
                                      transforms.ToTensor()
                                     ])

trainset = torchvision.datasets.CIFAR10(root ='./data', train = True, download = True, transform = test_transforms)
trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True)
testloader = DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)

mean = 0.
std = 0.
for images, _ in trainloader:
    batch_samples = images.size(0)
    images = images.view(np.array(batch_samples), images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)

mean /= len(trainloader)
std /= len(trainloader)

print(mean, std)

del trainset, trainloader, testset, testloader



"""## Data augmentation"""

train_transform = A.Compose(
    [
        A.HorizontalFlip(p = 0.5),
        A.ShiftScaleRotate(shift_limit = 0.05, scale_limit = 0.05, rotate_limit = 10, p = 0.5),
        A.CoarseDropout(max_holes = 1, max_height = 16, max_width = 16, min_holes = 1, 
                        min_height = 16, min_width = 1, 
                        fill_value = mean.tolist(), mask_fill_value = None),
        A.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)



"""### Custom data set to apply data augmentation"""

#custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, transform = None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)  # size of data

    def __getitem__(self, idx):
        record = self.data[idx]     # image
        img = np.array(record[0])   # converting to numpy array
        label = record[1]           # label

        if self.transform is not None:
            img = self.transform(image = img)["image"]    # applying transformations

        return img, label



"""### Train & test data loader"""

# creating data loaders
BATCH_SIZE = 16

traindata = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = None)
trainset = CustomDataset(traindata, transform = train_transform)
trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

testdata = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = None)
testset = CustomDataset(testdata, transform = test_transform)
testloader = DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



"""### Showing images"""

# showing image from a batch
images, labels = next(iter(testloader))
    
for i in range(BATCH_SIZE//2):
    plt.subplot(2, BATCH_SIZE//4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    imshow(images[i])
    plt.xlabel(classes[labels[i]])
plt.suptitle("Random examples")
plt.savefig('plots/random_examples.png')



"""### Creating model"""

# use gpu device and initiate the network
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = Net().to(device)
summary(model, input_size=(3, 32, 32))



"""### Loss, optimizer & learning rate scheduler"""

# Loss, optimizer & learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr = 0.001)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience = 3, factor = 1/3, min_lr = 1e-5)



"""### Training model"""

# storing losses & accuracy
test_losses = []
test_accs = []

EPOCHS = 60  # epochs

# training model
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(model, device, trainloader, optimizer, criterion, epoch)
    te_loss, te_acc = test(model, device, testloader, criterion)
    test_losses.append(te_loss)
    test_accs.append(te_acc)
    scheduler.step(te_acc)



"""### Model performance"""
print("\nFinal Model Performance: ")
evaluate(model, device, trainloader, criterion, split="Train")  # train performance
evaluate(model, device, testloader, criterion, split="Test")   # test performance



"""### Showing 10 mis-classified images"""

# extracting misclassified images
mis_pred = []
mis_true_label = []
mis_ex = []

model.eval()
with torch.no_grad():
     for data, target in testloader:
         data, target = data.to(device), target.to(device)
         output = model(data)
         pred = output.argmax(dim=1, keepdim=True)            # get the index of the max log-probability
         idx = (pred.view(-1) != target.view(-1))             # index of miss classification
         try:
            mis_pred.extend(pred[idx].squeeze().tolist())     # pred label
            mis_true_label.extend(target[idx].tolist())       # true label
            mis_ex.extend(data[idx].cpu())                    # image
         except:
            pass

# showing 10 mis-classified images
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    imshow(mis_ex[i])
    plt.xlabel(classes[mis_pred[i]])
plt.suptitle("Misclassified examples")
plt.savefig('plots/misclassified_images.png')

