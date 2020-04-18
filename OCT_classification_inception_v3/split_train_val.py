#从训练集中的每一类中随机抽取250张，移动到验证文件夹val对应的子类文件夹中
import os,random,shutil
path='OCT2017'
train_path='train'
val_path='val'
def joint(a1,a2):
    return os.path.join(a1,a2)
source=joint(path,train_path)
destination=joint(path,val_path)

def moveFile(dir_name):
    dir_s = joint(source, dir_name)
    dir_d=joint(destination, dir_name)
    os.makedirs(dir_d)#建立val文件夹下的子类文件夹
    rate = 0.1  # 自定义抽取图片的比例
    pathDir = os.listdir(dir_s)
    filenumber = len(pathDir)
    picknumber = int(filenumber * rate)
    # picknumber = 250#从训练集中的每一类中随机抽取250张
    pathDir = os.listdir(dir_s)
    # print(pathDir)#图片名称列表
    sample = random.sample(pathDir, picknumber)
    # 随机选取picknumber数量的样本图片
    print(sample)#打印出抽取的图片，可注释掉
    for name in sample:
        shutil.move(joint(dir_s,name), dir_d)
for dir in os.listdir(source):
    moveFile(dir)

