import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# loading data
data = np.loadtxt('data1.txt',delimiter=',')
print(data.shape)
num_feature = data.shape[1] - 1
data = data.astype('float32')
# data normalization
data_norm = data.copy()
# maximum = np.max(data_norm,axis=0,keepdims=True)
# print(maximum)
# minimun = np.min(data_norm,axis=0,keepdims=True)
# data_norm = (data_norm - minimun)/(maximum - minimun)#数据标准化
# print(data_norm)
# train val split
data_train, data_test = train_test_split(data_norm, test_size=0.3, random_state=42)
print(data_norm)

def data_normalization(data_train):
    data_max = np.max(data_train,axis=0,keepdims=True)
    data_min = np.min(data_train,axis=0,keepdims=True)
    data_train = (data_train - data_min)/(data_max - data_min)
    return data_train
data_train =  data_normalization(data_train)
data_test = data_normalization(data_test)
X_train = data_train[:, :2]
X_train = np.concatenate((X_train, np.ones((X_train.shape[0],1))), axis=1)
y_train = data_train[:, 2]
X_test = data_test[:, :2]
X_test = np.concatenate((X_test, np.ones((X_test.shape[0],1))), axis=1)
y_test = data_test[:, 2]
# model init
w = np.random.rand(num_feature+1,1)

# gradient descent
def L2_loss(y_pred,y):
    return np.mean(np.square(y_pred-y))

iterations = 5000
lr = 0.1
log = []
log_test = []
for i in range(iterations):
    y_pred = np.matmul(X_train, w)
    term = lr*np.mean((y_pred-y_train.reshape(-1,1))*X_train, axis=0).reshape(-1,1)
    w -= term
    loss = L2_loss(y_pred,y_train)
    print('iter:{},loss:{}'.format(i,loss))
    log.append([i,loss])
    y_pred = np.matmul(X_test, w)
    loss = L2_loss(y_pred,y_test)
    log_test.append([i,loss])


print("梯度下降次数： {},梯度下降求解：{},".format(iterations,w))

# normal eqution
term = np.matmul(X_train.T, X_train)
term_inv = np.linalg.inv(term)
w = np.matmul(np.matmul(term_inv,X_train.T),y_train.reshape(-1,1))
print("正规方程求解：{}".format(w))
# loss curve visualization
log = np.array(log)
plt.title("loss")
plt.plot(log[:,0],log[:,1])
log_test = np.array(log_test)
plt.plot(log_test[:,0],log_test[:,1])
plt.legend(["train_loss","test_loss"])
plt.show()

# visualization
y_pred = np.matmul(X_test, w)
plt.scatter(X_test[:,0],y_pred,c='r')
plt.scatter(X_test[:,0],y_test,c='b')
plt.show()

plt.scatter(X_test[:,1],y_pred,c='r')
plt.scatter(X_test[:,1],y_test,c='b')
plt.show()