from torch import nn
from torch.optim import Adam

from anomaly_detection.piad.utils import GradientNormHelper


class Optimizer(nn.Module):
    def __init__(self, enc_params, dec_params, edis_params, ddis_params,
                 image_adv_loss, latent_adv_loss, image_rec_loss,
                 latent_dim, adam_kwargs):
        super(Optimizer, self).__init__()

        self.adam_kwargs = adam_kwargs

        self.image_adv_loss = image_adv_loss
        self.latent_adv_loss = latent_adv_loss
        self.image_rec_loss = image_rec_loss

        self.rec_loss_weight_enc = 1
        self.rec_loss_weight_dec = 1
        self.grad_norm_helper = GradientNormHelper()

        self.latent_dim = latent_dim

        self.enc_dec_opt, self.ddis_opt, self.edis_opt = None, None, None

        self.set_new_params(enc_params, dec_params, edis_params,  ddis_params)

    def set_new_params(self, enc_params, dec_params, edis_params,  ddis_params, image_rec_loss=None):

        def preprocess(params):
            return [p for p in params if p.requires_grad]

        ddis_params = preprocess(ddis_params)
        edis_params = preprocess(edis_params)
        enc_dec_params = preprocess(enc_params) + preprocess(dec_params)

        self.ddis_opt = Adam(ddis_params, **self.adam_kwargs['ddis'])
        self.edis_opt = Adam(edis_params, **self.adam_kwargs['edis'])
        self.enc_dec_opt = Adam(enc_dec_params, **self.adam_kwargs['enc_dec'])

        if image_rec_loss is not None:
            self.image_rec_loss = image_rec_loss

    @staticmethod
    def compute_dis_loss(dis, loss, opt, real, fake, update_parameters):
        loss, loss_info = loss.dis_loss(dis, real.detach(), fake.detach())

        if update_parameters:
            opt.zero_grad()
            loss.backward()
            opt.step()

        return loss_info

    def compute_ddis_loss(self, ddis, real, fake, update_parameters=True):
        return self.compute_dis_loss(ddis, self.image_adv_loss, self.ddis_opt,
                                     real, fake,
                                     update_parameters=update_parameters)

    def compute_edis_loss(self, edis, real, fake, update_parameters=True):
        return self.compute_dis_loss(edis, self.latent_adv_loss, self.edis_opt,
                                     real, fake,
                                     update_parameters=update_parameters)

    @staticmethod
    def compute_adv_loss(dis, loss, real, rec):
        loss, loss_info = loss.gen_loss(dis, real, rec)
        return loss

    def compute_enc_dec_loss(self, enc, dec, edis, ddis, real_x, fake_x, fake_z, rec_x,
                             update_parameters=False, update_grad_norm=False):

        image_adv_loss = self.compute_adv_loss(ddis, self.image_adv_loss, None, fake_x)
        latent_adv_loss = self.compute_adv_loss(edis, self.latent_adv_loss, None, rec=fake_z)
        image_rec_loss = self.image_rec_loss(real_x, rec_x)

        "=========================================== compute total loss ==============================================="

        if update_grad_norm:
            for model_name, model, losses in [
                ('enc', enc, [('adv', latent_adv_loss), ('rec', image_rec_loss)]),
                ('dec', dec,  [('adv', image_adv_loss),  ('rec', image_rec_loss)])
            ]:
                for loss_name, loss in losses:
                    self.grad_norm_helper.update_grad_norm_dict(model, model_name, loss, loss_name)

            self.rec_loss_weight_enc = self.grad_norm_helper.get_loss_weight('enc', 'adv')['rec']
            self.rec_loss_weight_dec = self.grad_norm_helper.get_loss_weight('dec', 'adv')['rec']

        if update_parameters:
            if (image_rec_loss.item() > 1e6) or (latent_adv_loss > 1e6) \
                    or (image_adv_loss > 1e6):
                raise ValueError("Too large value of loss function (>10^6)!")

            self.enc_dec_opt.zero_grad()

            # 1. Calculate rec_x loss
            image_rec_loss.backward(retain_graph=True)

            # multiply by weight for encoder and generator
            for p in enc.parameters():
                if p.requires_grad:
                    p.grad *= self.rec_loss_weight_enc

            for p in dec.parameters():
                if p.requires_grad:
                    p.grad *= self.rec_loss_weight_dec

            # 2. Add other losses
            (latent_adv_loss + image_adv_loss).backward()
            self.enc_dec_opt.step()

        loss_info = {
            'image_adv_loss': image_adv_loss.item(),
            'latent_adv_loss': latent_adv_loss.item(),
            'image_rec_loss': image_rec_loss.item(),
            'rec_loss_weight_enc': self.rec_loss_weight_enc,
            'rec_loss_weight_dec': self.rec_loss_weight_dec
        }

        return loss_info
