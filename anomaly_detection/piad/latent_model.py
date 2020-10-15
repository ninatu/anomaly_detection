import torch


class LatentModel(torch.nn.Module):
    def __init__(self, latent_res, latent_dim):
        super(LatentModel, self).__init__()
        self.latent_res = latent_res
        self.latent_dim = latent_dim

    def sample(self, batch_size):
        z = torch.Tensor(torch.Size((batch_size,
                                     self.latent_dim,
                                     self.latent_res,
                                     self.latent_res))).normal_()
        return z
