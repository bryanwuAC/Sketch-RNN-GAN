import numpy as np
import torch
import utils
from os import listdir
from os.path import isfile, join
from hyperparameters import HyperParameters

if __name__ == '__main__':
    model_dir = "saved_models/"
    model_list = [f for f in listdir(model_dir) if isfile(join(model_dir, f))]
    for i in range(len(model_list)):
        model_name = model_list[i]
        print("Using saved model {} to generate image.".format(model_name))
        model = torch.load(join(model_dir, model_name))
        generator = model.generator
        N_max = model.N_max
        hps = model.hps

        dataset_name = utils.get_dataset_name_from_model_name(model_name)
        utils.generate_image_with_model(N_max, generator, dataset_name, hps.tau, hps.adv_loss_weight, num_images=50)