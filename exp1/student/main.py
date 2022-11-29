import os
import numpy as np
from dataloader import DataLoader
from components import FullyConnectLayer, ReluLayer, CrossEntropy


class MlpMnistModel(object):
    def __init__(self, input_size, hidden1, hidden2, out_size):
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_size = out_size

        # 初始化网络中各组件
        self.fc1 = FullyConnectLayer(self.input_size, self.hidden1)
        self.fc2 = FullyConnectLayer(self.hidden1, self.hidden2)
        self.fc3 = FullyConnectLayer(self.hidden2, self.out_size)
        self.relu1 = ReluLayer()
        self.relu2 = ReluLayer()

        # 定义需要更新参数的组件列表
        self.update_layer_list = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        # 前向传播流程
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        return x

    def backward(self, dloss):
        # 反向传播流程
        dh2 = self.fc3.backward(dloss)
        dh2 = self.relu2.backward(dh2)
        dh1 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh1)
        dh1 = self.fc1.backward(dh1)

    def step(self, lr):
        # 参数更新
        for layer in self.update_layer_list:
            layer.update_params(lr)

    def save_model(self, param_dir):
        # 保存权重和偏置
        params = {}
        params['w1'], params['b1'] = self.fc1.weight, self.fc1.bias
        params['w2'], params['b2'] = self.fc2.weight, self.fc2.bias
        params['w3'], params['b3'] = self.fc3.weight, self.fc3.bias
        np.save(param_dir, params)

    def load_model(self, params):
        # 加载权重和偏置
        self.fc1.load_params(params['w1'], params['b1'])
        self.fc2.load_params(params['w2'], params['b2'])
        self.fc3.load_params(params['w3'], params['b3'])


if __name__ == '__main__':
    # 设置
    mnist_npy_dir = 'mnist'
    epochs = 10
    batch_size = 64
    lr = 0.01
    print_freq = 100
    train_data_loader = DataLoader(mnist_npy_dir, batch_size=batch_size, mode='train')
    val_data_loader = DataLoader(mnist_npy_dir, batch_size=batch_size, mode='val')

    # 初始化模型
    model = MlpMnistModel(input_size=784, hidden1=128, hidden2=64, out_size=10)
    # 初始化损失函数
    criterion = CrossEntropy()

    best_loss = 999
    for idx_epoch in range(epochs):
        train_data_loader.shuffle_data()
        # 训练
        for id_1 in range(train_data_loader.batch_nums):
            train_data, train_labels = train_data_loader.get_data(id_1)
            # 前向传播
            output = model.forward(train_data)
            loss = criterion.forward(output, train_labels)
            # 反向传播
            dloss = criterion.backward(loss)
            model.backward(dloss)
            # 参数更新
            model.step(lr)

            if id_1 % print_freq == 0:
                print('Train Epoch %d, iter %d, loss: %.6f' % (idx_epoch, id_1, loss))

        # 验证
        mean_val_loss = []
        pred_results = np.zeros([val_data_loader.input_data.shape[0]])
        for id_2 in range(val_data_loader.batch_nums):
            val_data, val_labels = val_data_loader.get_data(id_2)
            prob = model.forward(val_data)
            val_loss = criterion.forward(prob, val_labels)
            mean_val_loss.append(val_loss)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[id_2 * val_labels.shape[0]:(id_2 + 1) * val_labels.shape[0]] = pred_labels

            if id_2 % print_freq == 0:
                print('Val Epoch %d, iter %d, loss: %.6f' % (idx_epoch, id_2, val_loss))

        accuracy = np.mean(pred_results == val_data_loader.input_label)
        mean_val_loss = np.array(mean_val_loss).mean()
        print('Val Epoch: %d, Loss: %d, Acc: %f' % (idx_epoch, mean_val_loss, accuracy))

        if mean_val_loss <= best_loss:
            best_loss = mean_val_loss
            if not os.path.exists('./ckpts'):
                os.makedirs('./ckpts')
            model.save_model(os.path.join('./ckpts', 'epoch_%d_loss_%.6f.npy' % (idx_epoch, mean_val_loss)))
