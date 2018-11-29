class HyperParameters:
    def __init__ (self):
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.dropout = 0.9
        ## Encoding vector dimension
        self.latent_vector_length = 128
        self.num_mixture = 20
        self.data_path = "sketchrnn_bee.npz"
        self.limit = 1000
        self.epsilon = 1e-5
        self.batch_size = 100
        ## Temperature hyperparameter
        self.tau = 0.4
        self.gan_loss_weight = 0.5
        self.num_epoch = 50000
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.lr = 0.001