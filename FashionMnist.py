import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import math
import numpy as np
batch_size = 64 #Batch：how many photos per group
num_epochs = 10
learning_rate = 0.3
dimension = 28
dropout = [0.0, 0.5, 0.7, 0.9, 0.99, 0.999]
# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
accu_table = []

# Training, augmented and testing datasets
train_augment_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor()
])

train_dataset = dsets.FashionMNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

train_aug_dataset = dsets.FashionMNIST(root='./data', 
                            train=True, 
                            transform=train_augment_transforms,
                            download=True)

test_dataset = dsets.FashionMNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True)

train_aug_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
# source code:
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_convolutional_neuralnetwork/
class CNNModel(nn.Module):
    def __init__(self, width, drop):
        super(CNNModel, self).__init__()
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected 1 (readout)
        kernel = 3
        kernel_max2 = 2
        stride1,stride2 = 1,1
        width_cnn1 = math.ceil((width-kernel+2)/stride1 + 1 )
        width_cnn2 = math.ceil((width_cnn1-kernel+2)/stride2 + 1 )
        width_max2 = math.ceil((width_cnn2-kernel_max2)/kernel_max2 + 1)
        self.fc1 = nn.Linear(32 * width_max2 * width_max2, 128)
        self.relu3 = nn.ReLU() 
        self.dropout = nn.Dropout(p=drop)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2 
        out = self.maxpool2(out)
        # Resize
        # Flattern the convolution output
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        out = self.relu3(out)
        #dropout
        out = self.dropout(out)
        #linear func
        out = self.fc2(out)
        return out

class Myclassifier:
  def __init__(self, dim, drop, rate):
    self.__model = CNNModel(dim, drop)
    self.__device = torch.device("cuda:0")
    self.__learning_rate = rate

  def Init(self):
    self.__model.to(self.__device)
    self.__criterion = nn.CrossEntropyLoss()
    self.__optimizer = torch.optim.SGD(self.__model.parameters(), lr=self.__learning_rate)
    # self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=self.__learning_rate, weight_decay=0.2)
    self.__scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.__optimizer, 'max', patience=2)

  def Run(self, num):
    train_losses, test_losses = [], []
    self.__model.train()
    for epoch in range(num_epochs):
      # running_loss += self.RunTrainLoaders(train_loader)
      running_loss = self.RunTrainLoaders(train_aug_loader)
      total, correct, test_loss= self.RunTestLoaders(test_loader)
      accuracy = 100 * correct // total
      train_losses.append(running_loss/len(train_aug_loader))
      test_losses.append(test_loss/len(test_loader))
      self.__scheduler.step(accuracy)
      if epoch == 9:
        accu_table.append(accuracy)   
      print('training loss: {}. test loss: {}. Accuracy: {}'.format(running_loss/len(train_aug_loader), test_loss/len(test_loader), accuracy))
      
    # plot loss for all 10 epochs
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.title("chenhsi_model_"+str(num))
    plt.legend(frameon=False)
    plt.show()
    #save model
    torch.save(self.__model, "chenhsi_model_"+str(num))

  def RunTrainLoaders(self, loader):
    running_loss = 0
    for i, (images, labels) in enumerate(loader):
      # Load images
      images = images.to(self.__device)
      labels = labels.to(self.__device)
      # Clear gradients w.r.t. parameters
      self.__optimizer.zero_grad()
      # Forward pass to get output/logits
      outputs = self.__model(images)
      # Calculate Loss: softmax --> cross entropy loss
      loss = self.__criterion(outputs, labels)
      # Getting gradients w.r.t. parameters
      loss.backward()
      # Updating parameters
      self.__optimizer.step()
      running_loss += loss.item()
    return running_loss