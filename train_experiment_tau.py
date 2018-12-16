from model import Model

if __name__ == '__main__':
    tau_list = [0.25, 0.5, 0.75]
    dataset_name = "sketchrnn_ambulance.npz"
    for experiment_tau in tau_list:
        model = Model(dataset_name[:-4])
        model.hps.tau = experiment_tau
        model.train_GAN()