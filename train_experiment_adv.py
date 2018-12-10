from model import Model

if __name__ == '__main__':
    adv_loss_weight_list = [0.25, 0.5, 0.75]
    dataset_name = "sketchrnn_bee.npz"
    for adv_loss_weight in adv_loss_weight_list:
        model = Model(dataset_name[:-4])
        model.hps.adv_loss_weight = adv_loss_weight
        model.train_GAN()