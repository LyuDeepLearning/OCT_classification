# 需要先运行split_train_val.py从train文件夹中划分出每类250个的验证集
'''目录说明：OCT2017目录下含test文件夹和train文件夹，predict存放待识别图片;
train、test文件夹下含四类文件夹;
其余文件均与OCT2017目录同级
通过运行split_train_val.py会在OCT2017目录下建立val文件夹，其中含cat和dog文件夹
'''
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import torch
from network import net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
path = "OCT2017"
train_path = 'train'
val_path = 'val'
test_path = 'test'
'''训练集需要进行数据集增强，故进行了额外操作'''
t = [transforms.RandomRotation(6),
     transforms.Resize(size=(230, 230)),
     transforms.RandomCrop(224)]
train_transform = transforms.Compose \
        ([

        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomApply(t, p=0.1),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

transform = transforms.Compose \
    ([transforms.Resize(size=(224, 224)),
      transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# [0.485, 0.456, 0.406],[0.229, 0.224, 0.225],Imagenet standards


train_orig = datasets.ImageFolder(root=os.path.join(path, train_path), transform=train_transform)
val_orig = datasets.ImageFolder(root=os.path.join(path, val_path), transform=transform)
test_orig = datasets.ImageFolder(root=os.path.join(path, test_path), transform=transform)
# ImageFolder假设所有的文件按文件夹保存好，每个文件夹下面存贮同一类别的图片，文件夹的名字为分类的名字
train_num = len(train_orig)
val_num = len(val_orig)


# 创建数据接口
def data_loader(train_img, batch_size, shuffle=True, num_workers=16):
    train_loader = DataLoader(train_img, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader


# 模型训练
def model(train_orig, val_orig, learn_rate=0.0009, num_epochs=32, batch_size=32):
    train_loader = data_loader(train_orig, batch_size=batch_size)
    val_loader = data_loader(val_orig, batch_size=batch_size, shuffle=False)

    network = net().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=learn_rate)
    cost_func = torch.nn.CrossEntropyLoss()  # 等价于log_softmax+nll_loss

    batches_train = train_num / batch_size
    batches_val = val_num / batch_size
    val_loss_min = 999.0
    val_best_acc = 0.0
    best_epoch = 0

    writer = SummaryWriter('log')
    for epoch in range(num_epochs):
        print("Epoch{}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        # keep track of training and validation loss each epoch
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        '''准备开始训练'''
        network.train()
        '''training loop'''
        for step, (batch_x, batch_y) in enumerate(train_loader):
            if step % 200 == 0:
                print(step)
            batch_x = batch_x.to(device)  # 使用gpu
            batch_y = batch_y.to(device)  # 使用gpu
            # 梯度归零
            optimizer.zero_grad()
            # 前向传播
            output = network(batch_x)
            # 计算成本
            cost = cost_func(output, batch_y)
            # 反向传播
            cost.backward()
            # 更新参数
            optimizer.step()
            train_loss += cost.item()
            _, predicted = torch.max(output.data, 1)
            # torch.max返回输出结果中，按dim=1行排列的每一行最大数据及他的索引，丢弃数据，保留索引
            train_acc += (predicted == batch_y).sum().item()
        train_loss /= batches_train
        train_acc /= train_num
        '''After training loops ends, start validation'''
        # Don't need to keep track of gradients
        with torch.no_grad():
            # Set to evaluation mode
            network.eval()
            # Validation loop
            for step, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.to(device)  # 使用gpu
                batch_y = batch_y.to(device)  # 使用gpu
                # 前向传播
                output = network(batch_x)
                # 计算成本
                cost = cost_func(output, batch_y)
                val_loss += cost.item()
                _, predicted = torch.max(output.data, 1)
                # torch.max返回输出结果中，按dim=1行排列的每一行最大数据及他的索引，丢弃数据，保留索引
                val_acc += (predicted == batch_y).sum().item()
        val_loss /= batches_val
        val_acc /= val_num

        # Save the model if validation loss decreases
        if val_acc > val_best_acc or (val_acc == val_best_acc and val_loss < val_loss_min):
            # Save model
            torch.save(network.state_dict(), "net_params.pkl")
            # Track improvement
            val_loss_min = val_loss
            val_best_acc = val_acc
            best_epoch = epoch
        # each epoch
        print(f'Training Loss: {train_loss:.4f} \t Validation Loss: {val_loss:.4f}')
        print(f'Training Accuracy: {100 * train_acc:.2f}% \t Validation Accuracy: {100 * val_acc:.2f}%')
        writer.add_scalars('CrossEntropyLoss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
        writer.add_scalars('accuracy', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)

    '''all epoch ends'''
    network.load_state_dict(torch.load('net_params_end.pkl'))
    print(f'\nBest epoch: {best_epoch} with loss: {val_loss_min:.2f} and acc: {100 * val_best_acc:.2f}%')
    writer.close()  # 程序推出前需要关闭writer，类似文件读写。

    '''PyTorch保存参数和加载的方法总结：
    #1 保存整个网络
    torch.save(net, PATH)
    #2 保存网络中的参数, 速度快，占空间少
    torch.save(net.state_dict(), PATH)
    #3 针对上面两种保存方法，加载的方法分别是：
    model_dict = torch.load(PATH)
    model_dict = model.load_state_dict(torch.load(PATH))
    '''


def accuracy(dataset, batchsize=32):
    network = net(False).to(device)
    network.load_state_dict(torch.load('net_params.pkl'))
    test_loader = data_loader(dataset, batchsize, shuffle=False, num_workers=1)

    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算
        network.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            # torch.max返回输出结果中，按dim=1行排列的每一行最大数据及他的索引，丢弃数据，保留索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 将预测及标签两相同大小张量逐一比较各相同元素的个数
            # .item()将tensor类别的int值转成python数字
    print(correct)
    print(total)
    print('the accuracy on test_set is {:.4f}'.format(correct / total))

#统计混淆矩阵
import numpy as np
def confusion_matrix(dataset, batchsize=16):
    network = net(False).to(device)
    network.load_state_dict(torch.load('net_params.pkl'))
    test_loader = data_loader(dataset, batchsize, shuffle=False, num_workers=4)
    matrix=np.zeros((4,4),dtype=np.int32)
    # print(matrix)
    with torch.no_grad():  # 关闭梯度计算
        network.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            # torch.max返回输出结果中，按dim=1行排列的每一行最大数据及他的索引，丢弃数据，保留索引
            for i in range(len(labels)):
                matrix[labels[i]][predicted[i]]+=1
    print(matrix)
if __name__ == '__main__':
    # model(train_orig, val_orig)  # 训练
    '''用训练好的模型统计测试集的准确率'''
    # accuracy(test_orig)
    confusion_matrix(test_orig)







