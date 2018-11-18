import numpy as np
import os
import cv2

class_map = {
    'Caterpillar': 0,
    'Dirty': 1,
    'Flat flower': 2,
    'Hole': 3,
    'Mold': 4,
    'pressing': 5,
    'Scratch': 6,
    'Spot': 7,
    'Warped': 8,
    'Zinc ash': 9,
    'Zinc residue': 10,
}

class_num = 11

root_dir = r'F:\data\augLib'
train_dir = r'F:\data\augLib\train'
test_dir = r'F:\data\augLib\test'


def _get_filelist(dir_path):
    file_list = []
    if not os.path.exists(dir_path):
        raise FileNotFoundError('dir_path is not exist!')
    for name in os.listdir(dir_path):
        file_list.append(os.path.join(dir_path,name))
    return file_list

def get_file(data_dir):
    img_list = []
    label_list = []
    for class_dir in _get_filelist(data_dir):
        if os.path.isdir(class_dir):
            # 注意，这里是windows下路径，最好统一成"/"
            class_dir = class_dir.replace('\\','/')
            class_name = class_dir.split('/')[-1]
            if class_name not in class_map:
                raise ValueError('class_map have no %s class' % class_name)
            filelist = _get_filelist(class_dir)
            # print(class_dir,len(filelist))
            img_list = img_list + filelist
            label_list = label_list \
                                + [class_map[class_name]]*len(filelist)
    onehot_label_list = [[float(i==label) for i in range(class_num)]for label in label_list]
    return img_list,onehot_label_list


def get_nextBatch(image, label, batch_size, step):
    start = batch_size * step
    end = batch_size * (step + 1)
    image_batch = image[start: end]
    label_batch = label[start: end]
    data_batch = []
    for img in image_batch:
        # print(img)
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        # img_resize = cv2.resize(img, (144, 144), interpolation=cv2.INTER_CUBIC)
        img_resize = img

        # data normalization
        img_resize = img_resize / np.max(img_resize)
        img_resize = img_resize - np.mean(img_resize)
        img_resize = img_resize.reshape(-1)

        data_batch.append(img_resize)

    data_batch = np.array(data_batch)
    label_batch = np.array(label_batch)

    return np.array(data_batch), np.array(label_batch)


if __name__ == '__main__':
    imglist,labellist = get_file(train_dir)
    a,b = get_nextBatch(imglist,labellist,5,20)

    print(a.shape)
    print(b.shape)
