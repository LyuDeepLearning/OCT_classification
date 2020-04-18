from torchvision import models
from torch import nn
import torch

# inceptionV3模型稍作修改，准备迁移学习
def net(pre=True):
    network = models.inception_v3(pretrained=False)#后面指定路径加载
    # print(network)
    network.aux_logits = False
    #Inception v3有2个输出，primary output和auxiliary output，一般只用到primary output
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
    path = 'inception_v3.pth'
    if pre == True:
        pretrained_dict = torch.load(path)
        net_dict = network.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict.items()}
        # print(pretrained_dict.items())
        # print(net_dict.items())
        # 2. overwrite entries in the existing state dict
        net_dict.update(pretrained_dict)
        network.load_state_dict(net_dict)
    para = [network.Mixed_7c, network.fc]
    for i in para:
        params=i.parameters()
        for param in params:
           param.requires_grad = True




    return network


if __name__ == '__main__':
    network = net()
    print(network)
