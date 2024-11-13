# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions

        # The first convolutional layer has 3 input channels and 6 output channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        # The second convolutional layer has 6 input channels and 16 output channels
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # The First fully connected layer has 16 * 5 * 5 input vectors and 256 output neurons
        self.fc1 = nn.Linear(16 * 5 * 5, 256)
        # The Second fully connected layer has 256 input vectors and 128 output neurons
        self.fc2 = nn.Linear(256, 128)
        # The Third fully connected layer has 128 input vectors and num_classes output neurons
        self.fc3 = nn.Linear(128, num_classes)


    def forward(self, x):
        shape_dict = {}
        # certain operations

        # After the first convolutional layer, change the negative values to 0
        x = torch.relu(self.conv1(x))
        # Max pooling to reduce the size of the image, and keep only important details
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        shape_dict[1] = list(x.shape)

        # After the second convolutional layer, change the negative values to 0
        x = torch.relu(self.conv2(x))
        # Max pooling to reduce the size of the image, and keep only important details
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        shape_dict[2] = list(x.shape)

        # Flatten the 3D tensor to a 1D to prepare for fully connected layers
        x = x.view(-1, 16 * 5 * 5)
        shape_dict[3] = list(x.shape)

        # Add non-linearity to the first fully connected layer
        x = torch.relu(self.fc1(x))
        shape_dict[4] = list(x.shape)

        # Add non-linearity to the second fully connected layer
        x = torch.relu(self.fc2(x))
        shape_dict[5] = list(x.shape)

        # Fully connected layer to get the final output
        x = self.fc3(x)
        shape_dict[6] = list(x.shape)

        return x, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet() # Initialize the LeNet model
    model_params = 0.0 # Initialize the variable to store the number of parameters

    # Loop through the model parameters to check the name and parameters
    for name, param in model.named_parameters():
        if param.requires_grad: # Check if the parameter requires gradient (i.e. is trainable)
            model_params += param.numel() # Add the number of elements in each parameters
            print(f"{name}: {param.numel()}")

    # Return the number of parameters in millions
    return model_params / 1e6


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
