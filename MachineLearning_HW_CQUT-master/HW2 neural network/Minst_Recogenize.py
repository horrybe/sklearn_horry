import numpy as np
import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            # The size of the picture is 28x28
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 14x14
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 7x7
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),


            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 64, out_features=128),
            torch.nn.ReLU(),

            nn.Dropout(0.6),
#             torch.nn.Linear(in_features=7 * 7 * 16, out_features=32),

            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, input):
        output = self.model(input)
        return output

device = "cuda:0" if torch.cuda.is_available() else "cpu"
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])

BATCH_SIZE = 1000

EPOCHS = 3
trainData = torchvision.datasets.MNIST('./data/', train=True, transform=transform, download=True)
testData = torchvision.datasets.MNIST('./data/', train=False, transform=transform)

trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = torch.utils.data.DataLoader(dataset=testData,batch_size=BATCH_SIZE)

net = Net()
torch.optim.SGD(net.parameters(),lr=0.01)
print(net.to(device))

lossF = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr = 0.001)
testLoss = 0
testAccuracy = 0
test_log_loss = []
test_log_Acc = []
history = {'Test Loss': [], 'Test Accuracy': []}

for epoch in range(1, EPOCHS + 1):
    processBar = tqdm(trainDataLoader, unit='step')
    net.train(True)
    for step, (trainImgs, labels) in enumerate(processBar):
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)


        net.zero_grad()
        outputs = net(trainImgs)
        loss = lossF(outputs, labels)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(predictions == labels) / labels.shape[0]
        loss.backward()
        optimizer.step()
        processBar.set_description("[%d/%d] Loss: %.10f, Acc: %.10f" %
                                   (epoch, EPOCHS, loss.item(), accuracy.item()))
        test_log_loss.append(loss.item())

        if step == len(processBar) - 1:
            correct, totalLoss = 0, 0
            net.train(False)
            with torch.no_grad():
                for testImgs, labels in testDataLoader:
                    testImgs = testImgs.to(device)
                    labels = labels.to(device)
                    outputs = net(testImgs)
                    loss = lossF(outputs, labels)
                    predictions = torch.argmax(outputs, dim=1)
                    totalLoss += loss
                    correct += torch.sum(predictions == labels)

                    testAccuracy = correct / (BATCH_SIZE*len(testDataLoader))
                    testLoss = totalLoss / len(testDataLoader)
                    history['Test Loss'].append(testLoss.cpu().numpy())
                    history['Test Accuracy'].append(testAccuracy.cpu().numpy())
            test_log_Acc.append(testAccuracy.cpu().numpy())
            processBar.set_description("[%d/%d] Loss: %.10f, Acc: %.10f, Test Loss: %.10f, Test Acc: %.10f" %(epoch, EPOCHS, loss.item(), accuracy.item(), testLoss, testAccuracy))
    processBar.close()

times_train = np.arange(test_log_loss.__len__())
times_test = np.arange(test_log_Acc.__len__())
plt.plot(times_train,test_log_loss, label='train Loss')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch*Bach_size')
plt.ylabel('Loss')
plt.show()

plt.plot(times_test,test_log_Acc, color='red', label='Test Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
