from torchvision import models
from torch import nn
import torch




# resnet模型稍作修改，准备迁移学习
def net(pre=True):
    # network = models.resnet50(pretrained=True)#用于获取预训练模型下载链接
    network = models.resnet50(pretrained=False)#后面指定路径加载
    # print(network)
    # Dropout(p=0.5, inplace=False)
    '''Keeping inplace=True will itself drop few values in the tensor input itself, 
    whereas if you keep inplace=False, you will to save the result of droput(input) 
    in some other variable to be retrieved.'''
    # 即inplace=True是原地进行操作，会更改输入；而False则会使用其他变量保存输出，不改变输入
    fc_inputs=network.fc.in_features
    network.fc=nn.Sequential(
        nn.Linear(fc_inputs,256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256,4)
    )
    pthfile = 'resnet50.pth'
    if pre == True:#pre==True表示加载预训练参数
        pretrained_dict = torch.load(pthfile)
        net_dict = network.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict.items()}
        # print(pretrained_dict.items())
        # print(net_dict.items())
        # 2. overwrite entries in the existing state dict
        net_dict.update(pretrained_dict)
        network.load_state_dict(net_dict)
    for param in network.parameters():
        param.requires_grad = False


    para = [network.layer3,network.layer4,network.avgpool, network.fc]
    for i in para:
        params = i.parameters()
        for param in params:
            param.requires_grad = True
    return network


if __name__ == '__main__':
    network = net()
    print(network)
