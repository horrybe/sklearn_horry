import torch
import matplotlib.pyplot as plt
import sklearn.model_selection as train_test_split
from scipy.io import loadmat

filename = 'data1.mat'
data_base = loadmat(filename)
data_now = []
for i in range(len(data_base["X"])):
    data_now.append([data_base["X"][i][0],data_base["X"][i][1],data_base["y"][i]])
data_train, data_test = train_test_split(data_now, test_size=0.3, random_state=45)
