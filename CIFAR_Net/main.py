import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from model.network import CIFAR_Net, DINOv2
from torchvision import transforms
import torchvision.datasets as datasets
from model.kernel import ConvModule, SKConvBlock
import os
import numpy as np

EPOCH_LIMIT = 25
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 16
SAVE_PATH = 'C:\\facultate\\prs\\prs_project\\models\\dino\\'

def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes


def test(model, test_loader, device, pad=False):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if pad:
                images = F.pad(images, (0, 10, 0, 10), mode='constant', value=0)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            images = images.cpu().numpy()
            labels = labels.cpu().numpy()
            predicted = predicted.cpu().numpy()

            for i in range(4):
                plt.subplot(1, 4, i + 1)
                plt.imshow(np.transpose(images[i], (1, 2, 0)))
                plt.title(f'Actual: {labels[i]}, Predicted: {predicted[i]}')
                plt.axis('off')

            plt.show()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            break


    accuracy = correct / total
    print('Test Accuracy: {:.2%}'.format(accuracy))
    return accuracy


def train_cifar_net():
    generate_plots = True
    train_loader, test_loader, classes = load_dataset() 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('training on device: {}'.format(device))

    net = CIFAR_Net(module=SKConvBlock, in_channels=3, dim_encoder=[16, 32, 64]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    train_losses_list = []
    test_accuracy_list = []
    train_accuracy_list = []
    for epoch in range(EPOCH_LIMIT):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f'Train Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}') 
        torch.save(net.state_dict(), "%s_epoch%d_loss%d.pkl" % (SAVE_PATH, epoch, int(running_loss / len(train_loader))))
        test_accuracy = test(net, test_loader, device)
        train_accuracy = test(net, train_loader, device)

        train_accuracy_list.append(train_accuracy)
        train_losses_list.append(running_loss / len(train_loader))
        test_accuracy_list.append(test_accuracy)
    
    if generate_plots:
        # Plot train loss
        plt.subplot(2, 1, 1)
        plt.plot(train_losses_list, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot test accuracy
        plt.subplot(2, 1, 2)
        plt.plot(test_accuracy_list, label='Test Accuracy', color='orange')
        plt.plot(train_accuracy_list, label='Train Accuracy', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
        # plt.savefig(SAVE_PATH + 'train_loss_test_accuracy.png')


def train_dino_net():
    generate_plots = True
    train_loader, test_loader, classes = load_dataset() 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('training on device: {}'.format(device))

    net = DINOv2().to(device)
    net.backbone.requires_grad_(False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    train_losses_list = []
    test_accuracy_list = []
    train_accuracy_list = []
    for epoch in range(EPOCH_LIMIT):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = F.pad(inputs, (0, 10, 0, 10), mode='constant', value=0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f'Train Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}') 
        torch.save(net.state_dict(), "%s_epoch%d_loss%d.pkl" % (SAVE_PATH, epoch, int(running_loss / len(train_loader))))
        test_accuracy = test(net, test_loader, device, pad=True)
        train_accuracy = test(net, train_loader, device, pad=True)

        train_accuracy_list.append(train_accuracy)
        train_losses_list.append(running_loss / len(train_loader))
        test_accuracy_list.append(test_accuracy)
    
    if generate_plots:
        # Plot train loss
        plt.subplot(2, 1, 1)
        plt.plot(train_losses_list, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot test accuracy
        plt.subplot(2, 1, 2)
        plt.plot(test_accuracy_list, label='Test Accuracy', color='orange')
        plt.plot(train_accuracy_list, label='Train Accuracy', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()

        plt.savefig(SAVE_PATH + 'train_loss_test_accuracy.png')


def visu_network():
    train_loader, test_loader, classes = load_dataset() 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CIFAR_Net(module=SKConvBlock, in_channels=3, dim_encoder=[16, 32, 64])
    net = torch.load('C:\\facultate\\prs\\prs_project\\models\\sk\\_epoch24_loss0.pkl')
    model.load_state_dict(net)
    model.to(device)
    test(model, test_loader, device, pad=False)


if __name__ == '__main__':
    # random_tesor = torch.rand(4, 3, 32, 32)
    # model = CIFAR_Net(module=ConvModule, in_channels=3, dim_encoder=[16, 32, 32])
    # # model = SKConvBlock(3, 
    # output = model(random_tesor)
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(pytorch_total_params)
    # print(output.shape)
    # torch.onnx.export(model, random_tesor, 'C:\\facultate\\prs\\prs_project\\models\\model-cfdsfdsfsdfdsffull2.onnx', input_names=['input'], output_names=['output'])
    # # train_cifar_net()
    visu_network()


