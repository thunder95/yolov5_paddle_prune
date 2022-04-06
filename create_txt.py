import os
import scipy.io as sio
from PIL import Image
import os, glob
import datetime
import shutil

def create_txt(dir_name, filename):
    with open(filename, "w") as txtfile:   # 在data文件夹下生成txt文件
        imglist = []
        imglist.extend(glob.glob(os.path.join(dir_name, "*.jpg")))   # img='images/test/abc.jpg'

        for idx, img in enumerate(imglist):
            if idx != 0:
                txtfile.write("\n")
            txtfile.write(img)    # 加上前缀data

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    ori_root="/home/aistudio/data/data127016/images"
    create_txt(os.path.join(ori_root, 'train2017'),'train.txt')
    create_txt(os.path.join(ori_root, 'val2017'), 'valid.txt')


