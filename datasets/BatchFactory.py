from datasets import Data_Augmentation_Gaoxian as gx
from datasets import Data_Augmentation_NEU as neu
from datasets import augLib
import random

datasets_map = {
    'gaoxian': gx,
    'NEU': neu,
    'augLib': augLib
}


def get_file(dataset_name,data_dir,is_shuffle = False):
    image, label = datasets_map[dataset_name].get_file(data_dir)
    if is_shuffle:
        temp = list(zip(image, label))
        random.shuffle(temp)
        image, label = zip(*temp)
    return list(image),list(label)


def get_next_batch(dataset_name,image,label,batch_size,step,):
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown: %s' % dataset_name)
    return datasets_map[dataset_name].get_nextBatch(image,label,batch_size,step)
