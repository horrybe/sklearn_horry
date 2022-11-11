import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# loading data
data = np.loadtxt('data2.txt',delimiter=',')
print(data.shape)
num_feature = data.shape[1] - 1

data = data.astype('float32')
# data normalization
data_ori = data.copy()

# train val split
data_train, data_test = train_test_split(data, test_size=0.3, random_state=45)
print(data_train,data_test)

def data_normalization(data_train):
    data_max = np.max(data_train,axis=0,keepdims=True)
    data_min = np.min(data_train,axis=0,keepdims=True)
    data_train = (data_train - data_min)/(data_max - data_min)
    return data_train
data_train = data_normalization(data_train)
data_test = data_normalization(data_test)
print(data_train,data_test)
X_train = data_train[:, :2]
X_train = np.concatenate((X_train, np.ones((X_train.shape[0],1))), axis=1)
y_train = data_train[:, 2].reshape(-1,1)
X_test = data_test[:, :2]
X_test = np.concatenate((X_test, np.ones((X_test.shape[0],1))), axis=1)
y_test = data_test[:, 2].reshape(-1,1)
# model init
w = np.zeros((num_feature+1,1))
R = 0.00001
def cross_entropy_loss(y_pred,y,w):
    return -np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred)) + np.mean((w**2)*R)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

iterations = 10000
lr = 0.6

log = []
log_test = []

# gradient descent
for i in range(iterations):
    y_pred = sigmoid(np.matmul(X_train, w))
    g = lr*(np.mean((y_pred-y_train)*X_train, axis=0).reshape(-1,1) + 2*w*R)
    w -= g
    loss = cross_entropy_loss(y_pred,y_train,w)
    print('iter:{},loss:{}'.format(i,loss))
    log.append([i,loss])

    y_pred_test = sigmoid(np.matmul(X_test, w))
    loss = cross_entropy_loss(y_pred_test,y_test,w)
    log_test.append([i,loss])
#     print('iter:{},val_loss:{}'.format(i,loss))

# loss curve visualization
log = np.array(log)

plt.plot(log[:,0],log[:,1])
plt.title("LOSS CURVE")
log_test = np.array(log_test)
plt.plot(log_test[:,0],log_test[:,1])
plt.legend(["loss_train_log","loss_test_log"])
plt.show()


# visualization
plt.scatter(X_train[:,0],X_train[:,1],c=y_train.flatten())
x = np.linspace(0,1,10)
y = (- w[0]*x - w[2])/w[1]
plt.plot(x, y)

plt.show()

plt.scatter(X_test[:,0],X_test[:,1],c=y_test.flatten())
x = np.linspace(0,1,10)
y = (- w[0]*x - w[2])/w[1]
plt.plot(x, y)
plt.show()

