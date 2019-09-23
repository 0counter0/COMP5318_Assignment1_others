import h5py
import numpy as np

import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

import random


with h5py.File('images_training.h5', 'r') as H:
    data = np.copy(H['data'])
with h5py.File('labels_training.h5', 'r') as H:
    label = np.copy(H['label'])
with h5py.File('images_testing.h5', 'r') as H:
    data_test = np.copy(H['data'])
with h5py.File('labels_testing_2000.h5', 'r') as H:
    label_validation = np.copy(H['label'])

'''
define the hyper-parameter:
weight_decay: for L2 norm regularization
batch_sz: batch size of each iteration
lr: learning rate
max_iter: the number of total iterations
'''
weight_decay = 0.5
batch_sz = 30
lr = 0.001
max_iter = 40000


class BP_network(object):

    def __init__(self):
        # 784 node in input layer kaiming initialization
        self.w1 = np.random.randn(784, 256) * np.sqrt(2 / 784)
        self.w2 = np.random.randn(256, 128) * np.sqrt(2 / 256)
        self.w3 = np.random.randn(128, 10) * np.sqrt(2 / 128)

        self.b1 = np.zeros([1, 256])
        self.b2 = np.zeros([1, 128])
        self.b3 = np.zeros([1, 10])

    def ReLu(self, x):
        '''

        :param x: the input of the ReLu function
        :return: ReLu(x)
        '''
        x[x < 0] = 0
        return x

    def softmax(self, x):
        '''
        :param x: the input of the softmax function
        :return: softmax(x)
        '''
        z = x.copy()
        result = np.zeros(z.shape)
        for i in np.arange(x.shape[0]):
            max_mzz = np.max(x[i])
            z[i] -= max_mzz
            result[i] = np.exp(z[i]) / np.sum(np.exp(z[i]))

        return result

    def forward_prop(self, input_data):
        '''

        :param input_data: the input data of the BP NN.

        the net is:
        x1 = ReLu(x0w1)
        x2 = ReLu(x1w2)
        x3 = softmax(x2w3)

        this is a void function, aiming to update the self.x3 and prepare for back propagation
        '''
        self.x0 = input_data.reshape(batch_sz, 784)  # (batch_sz , 784)
        self.x1 = self.ReLu(np.array(np.dot(self.x0, self.w1) + self.b1, dtype=np.float32))  # (batch_sz, 256)
        self.x2 = self.ReLu(np.array(np.dot(self.x1, self.w2) + self.b2, dtype=np.float32))
        self.x3 = self.softmax(np.array(np.dot(self.x2, self.w3) + self.b3, dtype=np.float32))  # (batch_sz , 10)

    def bp(self):
        '''

        this is a void function, aiming to update the weight matrix and bias
        '''

        d_na_x3 = self.x3.copy()
        d_na_x3[np.arange(batch_sz), input_label] -= 1

        dw3 = (self.x2.T.dot(d_na_x3) + (2 * weight_decay / batch_sz) * self.w3)

        db3 = np.sum(d_na_x3.copy(), axis=0)

        dx2 = d_na_x3.dot(self.w3.T)
        ReLu2 = self.x2.copy()
        ReLu2[ReLu2 < 0] = 0
        ReLu2[ReLu2 > 0] = 1
        d_na_x2 = dx2 * ReLu2
        db2 = np.sum(d_na_x2.copy(), axis=0)

        dw2 = (self.x1.T.dot(d_na_x2) + (2 * weight_decay / batch_sz) * self.w2)

        # dx1 = d_na_x2.dot(self.w2.T)
        dx1 = d_na_x2.dot(self.w2.T)

        # print(dx1)
        ReLu1 = self.x1.copy()
        ReLu1[ReLu1 < 0] = 0
        ReLu1[ReLu1 > 0] = 1
        d_na_x1 = dx1 * ReLu1
        db1 = np.sum(d_na_x1.copy(), axis=0)

        dw1 = (self.x0.T.dot(d_na_x1) + (2 * weight_decay / batch_sz) * self.w1)

        self.w1 -= lr * dw1
        self.w2 -= lr * dw2
        self.w3 -= lr * dw3
        self.b1 -= lr * db1
        self.b2 -= lr * db2
        self.b3 -= lr * db3

    def have_a_try(self, input_data, input_label=None):
        '''

        :param input_data:the data to be classify
        :param input_label:
            if is not None, then the input data is labeled
            else, the input data is the test set

        :return:
            if input_label is not None (the input data is labeled), return the accuracy and loss of prediction
            if input_label is None, then return the predict result.
        '''
        x1 = self.ReLu(np.dot(input_data, self.w1) + self.b1)
        x2 = self.ReLu(np.dot(x1, self.w2) + self.b2)
        x3 = self.softmax(np.dot(x2, self.w3) + self.b3)
        if input_label is not None:
            predict = np.argmax(x3, axis=1)  # the prediction
            s = predict - input_label  # compared with ground truth
            num = np.sum(1 for i in s if i == 0)  # calculate the number of correct prediction
            accuracy = num / input_data.shape[0]  # calculate the accuracy
            probability = x3[np.arange(input_data.shape[0]), input_label]
            Loss = -np.log(probability)
            return accuracy, Loss
        else:
            predict = np.argmax(x3, axis=1)  # the prediction
            return predict


if __name__ == '__main__':

    NN = BP_network()

    # define the train set and pre-process
    data = np.array(data, dtype=float)
    data = data.reshape(30000, -1)
    train_data = data
    train_label = label
    train_data -= np.mean(train_data, axis=0)
    train_data /= np.std(train_data, axis=0)

    # define the validation set, test set and pre-process
    data_test = np.array(data_test, dtype=float)
    data_test = data_test.reshape(5000, 784)
    data_test -= np.mean(data_test, axis=0)
    data_test /= np.std(data_test, axis=0)
    data_validation = data_test[0:2000]

    # define the x-axis and y-axis for plotting train loss, train acc, test loss, test acc VS epoch
    x = np.arange((batch_sz * max_iter) / train_data.shape[0])  # x-axis
    TL = np.array([])  # train loss
    TA = np.array([])  # train accuracy
    VL = np.array([])  # validation loss
    VA = np.array([])  # validation accuracy

    for iters in range(max_iter):
        # shuffle it at every beginning of each epoch
        if iters % (train_data.shape[0] / batch_sz) == 0:
            '''
            every (train_data.shape[0] / batch_sz) iterations, 
            the whole train set has been processed once, which was called an "epoch"
            '''
            inds = np.random.permutation(train_data.shape[0])
            train_data = train_data[inds]
            train_label = train_label[inds]
            '''
            the model begin converge around epoch 20, so adjust the learning rate manually
            '''
            if (1 + iters * batch_sz / train_data.shape[0]) in (20, 30):
                lr /= 10
            print("epoch: " + str(1 + iters * batch_sz / train_data.shape[0]) + " the lr: " + str(lr))

        # select the batch
        st_idx = int((iters % (train_data.shape[0] / batch_sz)) * batch_sz)
        ed_idx = st_idx + batch_sz
        input_data = train_data[st_idx: ed_idx].copy()
        input_label = train_label[st_idx: ed_idx].copy()

        # add gaussian noise
        for i in np.arange(input_data.shape[0]):
            input_data[i] += random.gauss(0, 0.01)

        # forward propagation

        NN.forward_prop(input_data)
        NN.bp()

        if iters % (train_data.shape[0] / batch_sz) == 0:
            # for every epoch, we output the train acc and loss
            t_accuracy, t_Loss = NN.have_a_try(train_data, train_label)
            TA = np.append(TA, t_accuracy)
            TL = np.append(TL, (np.mean(np.abs(t_Loss))))
            print("train accuracy: " + str(t_accuracy) + "  train loss: " + str((np.mean(np.abs(t_Loss)))))

            # for every epoch, we output the validation acc and loss
            v_accuracy, v_Loss = NN.have_a_try(data_validation, label_validation)
            VA = np.append(VA, v_accuracy)
            VL = np.append(VL, (np.mean(np.abs(v_Loss))))
            print("test accuracy : " + str(v_accuracy) + "  test Loss: " + str(np.mean(np.abs(v_Loss))))

    plt.ylim(0, 2)
    plt.subplot(211)
    plt.plot(x, TA, x, TL)
    plt.subplot(212)
    plt.plot(x, VA, x, VL)
    plt.show()

    test_prediction = NN.have_a_try(data_test)
    with h5py.File('predicted_labels.h5', 'w') as H:
        H.create_dataset('label', data=test_prediction)
    with h5py.File('predicted_labels.h5', 'r') as H:
        mzz = np.copy(H['label'])
    print(mzz)
    print(mzz.shape)
