
# OCT_classification project  
https://github.com/LyuDeepLearning/OCT_classification  
PyTorch project for OCT images classification based on transfer learning  
[VGG19, InceptionV3, ResNet50]  

1.下载数据集:http://dx.doi.org/10.17632/rscbjbr9sj.2  
  下载公开数据集OCT2017.tar.gz,放在和split_train_val.py同目录下，解压，得到OCT2017文件夹  
  OCT2017数据集（四种类别）包含train和test,所以需要自行划分验证集val  
2.运行split_train_val.py,从train中每一类抽取250张作为验证集  
3.下载预训练模型    
vgg19_bn:  
https://download.pytorch.org/models/vgg19_bn-c79401a0.pth  
下载后将文件名改为vgg19_bn.pth(含后缀)，放到OCT_classification_vgg19文件夹下  

inception_v3:    
https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth    
下载后将文件名改为inception_v3.pth(含后缀)，放到OCT_classification_inception_v3文件夹下  

resnet50:  
https://download.pytorch.org/models/resnet50-19c8e357.pth  
下载后将文件名改为resnet50.pth(含后缀)，放到OCT_classification_resnet50文件夹下  

4.以OCT_classification_resnet50为例:（其余两个模型类似）  
运行TransferLearning.py即可训练模型  
运行predict.py即可使用训练好的模型来预测指定文件夹中的图片的类别  




