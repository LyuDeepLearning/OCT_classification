import torch
from classes import index_class
from network import net
from PIL import Image
import numpy as np
import os
from torchvision import datasets,transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1.图片的预处理定义
transform = transforms.Compose\
    ([transforms.Resize(size=(229,229)),
      transforms.ToTensor(),# range [0, 255] -> [0.0,1.0]
      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])] )



#2.加载网络及参数
network =net(False).to(device)
network.load_state_dict(torch.load('net_params.pkl'))
print("OCT影像识别系统加载成功")


#3预测
def predict(x):
    x = np.expand_dims(x, axis=0)
    with torch.no_grad():
       network.eval()
       output = network(torch.from_numpy(x).to(device).float())
       pred_Y = torch.max(output, dim=1)[1].data.cpu().numpy()
       index=np.squeeze(pred_Y)
       return index_class(index)

import tkinter as tk
from tkinter import filedialog
import time

root = tk.Tk()
root.wm_attributes('-topmost',1)
root.withdraw()
while(True):
 print('------------------------')
 predict_path=''
 predict_path = filedialog.askdirectory()
 if predict_path=='':
     break
 print('您选取了文件夹'+predict_path)
 start = time.time()
 for pic_name in os.listdir(predict_path):
        pic_path=os.path.join(predict_path,pic_name)
        img=Image.open(pic_path)
        img=img.convert('RGB')
        x=transform(img)
        print("{}的预测类别为{}".format(pic_name,predict(x)))
 end = time.time()
 running_time = end - start
 print('总共识别' + str(len(os.listdir(predict_path))) + '张OCT影像' )
 print('耗时: %.5f 秒' % running_time)
 print('文件夹'+predict_path+'识别完毕')

print('感谢您的使用！')
