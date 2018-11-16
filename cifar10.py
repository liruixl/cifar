import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    path = r'F:\data\cifar-10-batches-py\data_batch_1'
    cifar10_dict = unpickle(path)
    # print(cifar10_dict['data'][0],cifar10_dict['labels'][0])
    print(cifar10_dict.keys())
    img1 = cifar10_dict[b'data'][58]
    print(type(img1))
    print(img1.shape)
    print(img1.dtype)
    img = np.reshape(img1,(3,32,32))
    img1 = np.stack([img[0],img[1],img[2]],-1)
    img2 = img.transpose(1, 2, 0)

    print(img1.shape)
    print(img2.shape)

    plt.imshow(img1)  # 必须规定为显示的为什么图像
    plt.show()  # 显示出来，不要也可以，但是一般都要了
    plt.imshow(img2)
    plt.show()  # 显示出来，不要也可以，但是一般都要了

    img3 = np.stack([img[2],img[1],img[0]],-1)
    cv2.imshow('first',img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()