from abc import ABC, abstractmethod
from enum import Enum
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from anomaly_detection.dpa.losses import wasserstein_loss, gradient_penalty, mse_loss, compute_grad2


class AbstractAdversarialLoss(ABC):
    @abstractmethod
    def gen_loss(self, dis, real_x=None, fake_x=None):
        """
        :return: (loss, loss_info), where
            loss is Tensor,
            loss_info is a dict {'subloss_name': subloss_value, ...} for logging. By default empty dict
        """
        pass

    @abstractmethod
    def dis_loss(self, dis, real_x=None, fake_x=None):
        """
        :return: (loss, loss_info), where
            loss is Tensor,
            loss_info is a dict {'subloss_name': subloss_value, ...} for logging. By default empty dict
        """
        pass

    @abstractmethod
    def dis_gen_loss(self, dis, real_x=None, fake_x=None):
        """
        :return: ((gen_loss, gen_loss_info), (dis_loss, dis_loss_info))
        """
        pass


class WassersteinLoss(AbstractAdversarialLoss):
    def __init__(self, lambd, gradient_penalty, norm_penalty=0, target_gamma=1.0):
        self._lambd = lambd
        self._gradient_penalty = gradient_penalty
        self._norm_penalty = norm_penalty
        self._target_gamma = target_gamma

    def gen_loss(self, dis, real_x=None, fake_x=None):
        assert fake_x is not None
        loss = wasserstein_loss(dis, x=real_x, tilda_x=fake_x)
        return loss, {'loss': loss.data.item()}

    def dis_loss(self, dis, real_x=None, fake_x=None):
        wass_loss, norm = wasserstein_loss(dis, x=real_x, tilda_x=fake_x, return_norm=True)
        norm = self._norm_penalty * norm
        wass_loss = - self._lambd * wass_loss

        with torch.set_grad_enabled(True):
            gp_loss = self._gradient_penalty * gradient_penalty(dis, real_x, fake_x, self._target_gamma)
        total = wass_loss + norm + gp_loss

        loss_info = {
            'wass': wass_loss.data.item(),
            'norm': norm.data.item(),
            'gp': gp_loss.data.item(),
        }

        return total, loss_info

    def dis_gen_loss(self, dis, real_x=None, fake_x=None):
        wass_loss, norm = wasserstein_loss(dis, x=real_x, tilda_x=fake_x, return_norm=True)

        gen_total = wass_loss
        gen_loss_info = {'loss': gen_total.data.item()}

        norm = self._norm_penalty * norm
        dis_wass_loss = - self._lambd * wass_loss

        with torch.set_grad_enabled(True):
            gp_loss = self._gradient_penalty * gradient_penalty(dis, real_x, fake_x, self._target_gamma)

        dis_total = dis_wass_loss + norm + gp_loss
        dis_loss_info = {
            'wass': dis_wass_loss.data.item(),
            'norm': norm.data.item(),
            'gp': gp_loss.data.item(),
        }

        return (dis_total, dis_loss_info), (gen_total, gen_loss_info)


class LsLoss(AbstractAdversarialLoss):
    def __init__(self, use_mulnoise=False):
        self._use_mulnoise = use_mulnoise
        self._mean_exp_output = None

    def gen_loss(self, dis, real_x=None, fake_x=None):
        loss = mse_loss(dis, x=fake_x)
        return loss, {'loss': loss.data.item()}

    def dis_loss(self, dis, real_x=None, fake_x=None):
        loss, cur_mean_output = mse_loss(dis, real_x, fake_x, return_mean_output=True)

        if self._use_mulnoise:
            if self._mean_exp_output is None:
                strength = dis.strength
                self._mean_exp_output = (strength / 0.2) + 0.5

            self._mean_exp_output = 0.1 * cur_mean_output + 0.9 * self._mean_exp_output
            strength = 0.2 * max(0, self._mean_exp_output - 0.5)
            dis.set_strength(strength)

        return loss, {'loss': loss.data.item()}

    def dis_gen_loss(self, dis, real_x=None, fake_x=None):
        raise NotImplementedError()


class NNL(AbstractAdversarialLoss):
    def __init__(self, use_reg=False, reg_alpha=0, reg_type='real'):
        assert reg_type in ['real', 'fake', 'real_fake']
        self.use_reg = use_reg
        self.reg_alpha = reg_alpha
        self.reg_type = reg_type

    def gen_loss(self, dis, real_x=None, fake_x=None):
        loss = 0
        if real_x is not None:
            dis_x = dis(real_x)
            loss += binary_cross_entropy_with_logits(dis_x, torch.zeros_like(dis_x, requires_grad=True))
        if fake_x is not None:
            dis_x = dis(fake_x)
            loss += binary_cross_entropy_with_logits(dis_x, torch.ones_like(dis_x, requires_grad=True))
        return loss, {'loss': loss.data.item()}

    def dis_loss(self, dis, real_x=None, fake_x=None):
        loss = 0
        reg_real = torch.tensor([0.0], requires_grad=True)
        reg_fake = torch.tensor([0.0], requires_grad=True)

        if real_x is not None:
            real_x.requires_grad_(True)

            dis_x = dis(real_x)
            loss += binary_cross_entropy_with_logits(dis_x, torch.ones_like(dis_x, requires_grad=True))

            if self.use_reg and (self.reg_type == 'real' or self.reg_type == 'real_fake'):
                # print(dis_x.re)
                reg_real = self.reg_alpha * compute_grad2(dis_x, real_x).mean()
        if fake_x is not None:
            dis_x = dis(fake_x)
            loss += binary_cross_entropy_with_logits(dis_x, torch.zeros_like(dis_x, requires_grad=True))

            if self.use_reg and (self.reg_type == 'fake' or self.reg_type == 'real_fake'):
                reg_fake = self.reg_alpha * compute_grad2(dis_x, fake_x).mean()

        reg_real = reg_real.to(loss.device)
        reg_fake = reg_fake.to(loss.device)

        return loss + reg_real + reg_fake, {
            'loss': loss.item(),
            'reg_real': reg_real.item(),
            'reg_fake': reg_fake.item(),
        }

    def dis_gen_loss(self, dis, real_x=None, fake_x=None):
        raise NotImplementedError()


class AdversarialLossType(Enum):
    wasserstein = 'wasserstein'
    least_squared = 'least_squared'
    nnl = 'nnl'


ADVERSARIAL_LOSSES = {
    AdversarialLossType.wasserstein: WassersteinLoss,
    AdversarialLossType.least_squared: LsLoss,
    AdversarialLossType.nnl: NNL
}


