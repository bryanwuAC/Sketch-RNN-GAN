from model import Model
from os import listdir
from os.path import isfile, join


if __name__ == '__main__':
    dataset_dir = "dataset/"
    dataset_list = [f for f in listdir(dataset_dir) if isfile(join(dataset_dir, f))]
    for dataset_name in dataset_list:
        model = Model(dataset_name[:-4])
        model.train_RNN()