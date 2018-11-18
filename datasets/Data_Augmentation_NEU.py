import os
import cv2
import numpy as np
import random
from skimage import io,util

# 数据增强方式：一张图四个角落和中心裁出五张，再水平翻转
root_dir = 'F:/data/NEU/NEU-CLS/'
train_dir = 'F:/data/NEU/train/'
test_dir = 'F:/data/NEU/test/'
defect_class = {'Cr':0,'In':1,'Pa':2,'PS':3,'RS':4,'Sc':5}


def getAllImg(root_dir):
    trainList, testList = [], []
    defect_Cr, defect_In, defect_Pa, defect_PS, defect_RS, defect_Sc = [],[],[],[],[],[]
    for file in os.listdir(root_dir):
        if 'Cr' in file:
            defect_Cr.append(file)
        elif 'In' in file:
            defect_In.append(file)
        elif 'Pa' in file:
            defect_Pa.append(file)
        elif 'PS' in file:
            defect_PS.append(file)
        elif 'RS' in file:
            defect_RS.append(file)
        elif 'Sc' in file:
            defect_Sc.append(file)
    random.shuffle(defect_Cr)
    random.shuffle(defect_In)
    random.shuffle(defect_Pa)
    random.shuffle(defect_PS)
    random.shuffle(defect_RS)
    random.shuffle(defect_Sc)
    trainList= defect_Cr[0:150] + defect_In[0:150] + defect_Pa[0:150] + defect_PS[0:150] + defect_RS[0:150] + defect_Sc[0:150]
    testList = defect_Cr[150:300] + defect_In[150:300] + defect_Pa[150:300] + defect_PS[150:300] + defect_RS[150:300] + defect_Sc[150:300]

    return trainList,testList


def augmentation(trainList, testList, root_dir):
    for file in trainList:
        image = root_dir + file
        imageName = file.split('.')[0]
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        # 裁剪
        img_cropped1 = img[0:166, 0:166]
        img_cropped2 = img[0:166, 34:200]
        img_cropped3 = img[34:200, 0:166]
        img_cropped4 = img[34:200, 34:200]
        img_cropped5 = img[17:183, 17:183]

        # 水平翻转
        img_flapped1 = cv2.flip(img_cropped1, 1, dst=None)
        img_flapped2 = cv2.flip(img_cropped1, 1, dst=None)
        img_flapped3 = cv2.flip(img_cropped1, 1, dst=None)
        img_flapped4 = cv2.flip(img_cropped1, 1, dst=None)
        img_flapped5 = cv2.flip(img_cropped1, 1, dst=None)

        # 均值模糊和高斯模糊
        Blur = cv2.blur(img, (5, 5), 3)
        Gussian_blur = cv2.GaussianBlur(img, (5, 5), 3)

        # 旋转角度
        RotateMatrix1 = cv2.getRotationMatrix2D(center=(img.shape[1]/2, img.shape[0]/2),angle=90,scale=1)
        RotImg_90 = cv2.warpAffine(img,RotateMatrix1,(img.shape[0],img.shape[1]))

        RotateMatrix2 = cv2.getRotationMatrix2D(center=(img.shape[1]/2, img.shape[0]/2),angle=180,scale=1)
        RotImg_180 = cv2.warpAffine(img, RotateMatrix2, (img.shape[0], img.shape[1]))

        RotateMatrix3 = cv2.getRotationMatrix2D(center=(img.shape[1]/2, img.shape[0]/2),angle=270,scale=1)
        RotImg_270 = cv2.warpAffine(img, RotateMatrix3, (img.shape[0], img.shape[1]))

        gaussian_nosie = util.random_noise(img, mode='gaussian', clip=True)
        poisson_nosie = util.random_noise(img, mode='poisson', clip=True)

        img_aug = [img_cropped1, img_cropped2, img_cropped3, img_cropped4, img_cropped5,
                   img_flapped1, img_flapped2, img_flapped3, img_flapped4, img_flapped5,
                   Blur, Gussian_blur, RotImg_90, RotImg_180, RotImg_270]

        for i in range(len(img_aug)):
            fileName = train_dir + imageName + '_aug_' + str(i + 1) + '.bmp'
            img_aug[i] = cv2.resize(img_aug[i],(144,144),interpolation=cv2.INTER_CUBIC)
            # print(img_aug[i].shape)
            cv2.imwrite(fileName, img_aug[i])


    for file in testList:
        image = root_dir + file
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        fileName = test_dir + file
        img = cv2.resize(img,(144,144),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(fileName, img)

def get_file(data_dir):
    fileList, labelList = [], []
    for file in os.listdir(data_dir):
        if '.bmp' in file:
            index = defect_class[file.split('_')[0]]
            # 转换成one_hot编码
            label = [0 for i in range(len(defect_class))]
            label[index] = 1
            fileList.append(data_dir+file)
            labelList.append(index)

    temp = np.array([fileList,labelList])
    temp = temp.transpose()
    np.random.shuffle(temp)

    img_list = list(temp[:,0])
    label_list = list(temp[:,1])
    onehot_label = []
    for i in label_list:
        label = [0 for j in range(len(defect_class))]
        label[int(i)] = 1
        onehot_label.append(label)

    return img_list, onehot_label

def get_nextBatch(image, label, batch_size, step):
    start = batch_size * step
    end = batch_size * (step + 1)
    image_batch = image[start : end]
    label_batch = label[start : end]
    data_batch = []
    for img in image_batch:
        # print(img)
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img_resize = cv2.resize(img, (144, 144), interpolation=cv2.INTER_CUBIC)

        # data normalization
        img_resize = img_resize / np.max(img_resize)
        img_resize = img_resize - np.mean(img_resize)
        img_resize = img_resize.reshape(-1)

        data_batch.append(img_resize)

    data_batch = np.array(data_batch)
    label_batch = np.array(label_batch)

    return np.array(data_batch), np.array(label_batch)



if __name__ == '__main__':
    # image, label = get_file(train_dir)
    # data_batch, label_batch = get_nextBatch(image, label, 20, 0)
    # print(data_batch)

    trainList, testList = getAllImg(root_dir)
    print(trainList)
    augmentation(trainList, testList, root_dir)

    print('Hello,tsy!')







