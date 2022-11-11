import os.path

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
BATCH_SIZE = 100
root = "data/MNIST/raw"
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
trainData = torchvision.datasets.MNIST('./data/', train=True, transform=transform, download=True)
testData = torchvision.datasets.MNIST('./data/', train=False, transform=transform)

trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE)
train = (torchvision.datasets.mnist.read_image_file(os.path.join(root,'train-images-idx3-ubyte')),
         torchvision.datasets.mnist.read_label_file(os.path.join(root,'train-labels-idx1-ubyte')))

counter = 0

for i in range(100):
    counter+=1
    test1 = train[0][i]
    test2 = train[1][i]
    plt.subplot(10,10,counter)
    plt.imshow(test1)
plt.show()