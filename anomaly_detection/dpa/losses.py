import torch
import torch.autograd as autograd
import numpy as np


def wasserstein_loss(dis, x=None, tilda_x=None, return_norm=False):
    loss = 0
    norm = 0
    if x is not None:
        dis_x = dis(x)
        loss += dis_x.mean()
        if return_norm:
            norm += (dis_x ** 2).mean()
        del dis_x
    if tilda_x is not None:
        dis_tilda_x = dis(tilda_x)
        loss -= dis_tilda_x.mean()
        if return_norm:
            norm += (dis_tilda_x ** 2).mean()
        del dis_tilda_x
    if return_norm:
        return loss, norm
    else:
        return loss


def mse_loss(dis, x=None, tilda_x=None, return_mean_output=False, smooth_labels=False):
    mse = torch.nn.MSELoss()
    loss = []
    output = []
    if x is not None:
        pred_label = dis(x)
        if smooth_labels:
            real_label = autograd.Variable(torch.Tensor(pred_label.size()).uniform_(0.8, 1.2).cuda())
        else:
            real_label = autograd.Variable(torch.Tensor(pred_label.size()).fill_(1).cuda())
        loss.append(mse(pred_label, real_label))
        output.extend(pred_label.data.cpu().numpy().tolist())
        del real_label, pred_label
    if tilda_x is not None:
        pred_label = dis(tilda_x)
        if smooth_labels:
            fake_label = autograd.Variable(torch.Tensor(pred_label.size()).uniform_(-0.2, 0.2).cuda())
        else:
            fake_label = autograd.Variable(torch.Tensor(pred_label.size()).fill_(0).cuda())
        loss.append(mse(pred_label, fake_label))
        output.extend(pred_label.data.cpu().numpy().tolist())
        del fake_label, pred_label
    if return_mean_output:
        mean_output = np.array(output).mean()
        return sum(loss) / len(loss), mean_output
    else:
        return sum(loss) / len(loss)


def gradient_penalty(dis, x, tilda_x, target_gamma=1.0):
    batch_size = x.size(0)
    shape = [x.shape[0]] + [1] * (len(x.shape) - 1)
    alpha = torch.rand(batch_size, 1).view(*shape).cuda()
    hat_x = alpha * x.data + ((1 - alpha) * tilda_x.data)
    hat_x = hat_x.detach().requires_grad_(True)

    hat_y = dis(hat_x)

    gradients = autograd.grad(outputs=hat_y,
                              inputs=hat_x,
                              grad_outputs=torch.ones(hat_y.size()).cuda(),
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty_value = (((gradients.norm(2, dim=1) - target_gamma) ** 2) / (target_gamma ** 2)).mean()
    return gradient_penalty_value


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True,
        only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg
