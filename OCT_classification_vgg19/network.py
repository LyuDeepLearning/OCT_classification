from torchvision import models
from torch import nn
import torch


# VGG16模型稍作修改，准备迁移学习
def net(pre=True):
    # network = models.vgg19_bn(pretrained=True)#用于获取预训练模型下载链接
    network = models.vgg19_bn(pretrained=False)
    # print(network)
    # Dropout(p=0.5, inplace=False)
    '''Keeping inplace=True will itself drop few values in the tensor input itself, 
    whereas if you keep inplace=False, you will to save the result of droput(input) 
    in some other variable to be retrieved.'''
    # 即inplace=True是原地进行操作，会更改输入；而False则会使用其他变量保存输出，不改变输入
    network.classifier = torch.nn.Sequential(
        nn.Linear(in_features=25088, out_features=1024),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1024, out_features=4))
    path='vgg19_bn.pth'
    if pre==True:#pre==True表示加载预训练参数
        pretrained_dict=torch.load(path)
        net_dict=network.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict.items()}
        # print(pretrained_dict.items())
        # print(net_dict.items())
        # 2. overwrite entries in the existing state dict
        net_dict.update(pretrained_dict)
        network.load_state_dict(net_dict)


    for param in network.parameters():
        param.requires_grad = False
    para=[]
    for i in range(-7,-1):
        para.append(network.features[i])
    # para.extend( [ network.avgpool,network.classifier])
    para.append( network.classifier)
    for i in para:
        params = i.parameters()
        for param in params:
            param.requires_grad = True

    return network


if __name__ == '__main__':
    network = net()
    print(network)
