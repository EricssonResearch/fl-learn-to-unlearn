"""Loss definitions."""
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

import ltu


EPS = torch.finfo(torch.float32).eps

LOG_PATH = os.path.join(ltu.__path__[0][0:-3], 'log')


class UnlearnLoss(nn.Module):
    """Blocks of loss functions.
    Implemented:
        reconstruction loss: MSE loss with regularization l_rec(x1, x2) = ||x1 - x2||^2 / ||x1||^2
        regularization loss: MSE cosine similarity loss, can be reduced to
        l_reg(x1, x21, ..., x2N) = -1 + 1/N (cos(x1, x21) + ... + cos(x1, x2N))
    """

    def __init__(self, sigmoid: bool = False) -> None:
        self.sigmoid = sigmoid
        super().__init__()

    def loss_rec(self, x_input, x_target):
        """x_target is the reference and x_input is the inferred input.
        """
        if self.sigmoid:
            x_input = torch.sigmoid(x_input)
            x_target = torch.sigmoid(x_target)
        mse_loss = nn.MSELoss(reduction='sum')
        return mse_loss(input=x_input, target=x_target) / (x_target.norm() + EPS)

    def loss_reg1(self, x_input_list, target_id):
        """Regularization loss."""
        n_agents = x_input_list.shape[1]
        ids = list(range(n_agents))
        ids.remove(target_id)
        cos = nn.CosineSimilarity(dim=1, eps=EPS)
        norm_coef = len(ids)
        x_target = x_input_list[:, target_id, :]
        cos_sum = 0
        if self.sigmoid:
            for i in ids:
                arg = x_input_list[:, i, :]
                cos_sum += torch.sum(cos(torch.sigmoid(x_target), torch.sigmoid(arg)))
        else:
            for i in ids:
                arg = x_input_list[:, i, :]
                cos_sum += torch.sum(cos(x_target, arg))
        return -1 + (1/norm_coef) * cos_sum

    def loss_reg2(self, x_input_list, x_target):
        """Regularization loss."""
        cos = nn.CosineSimilarity(dim=1, eps=EPS)
        norm_coef = x_input_list.shape[1]
        cos_sum = 0
        if self.sigmoid:
            for i in range(norm_coef):
                arg = x_input_list[:, i, :]
                cos_sum += torch.sum(cos(torch.sigmoid(x_target), torch.sigmoid(arg)))
        else:
            for i in range(norm_coef):
                arg = x_input_list[:, i, :]
                cos_sum += torch.sum(cos(x_target, arg))
        return -1 + (1/norm_coef) * cos_sum

    def forward(
        self,
        input_theta: Tensor,
        target_theta: Tensor,
        input_z: Tensor,
        target_z: Tensor,
        input_t: Tensor,
        input_theta_tilde_agent: Tensor,
        target_theta_agent
    ) -> Tensor:
        """Forward method."""
        rec_loss_theta = self.loss_rec(x_input=input_theta, x_target=target_theta)
        rec_loss_z = self.loss_rec(x_input=input_z, x_target=target_z)
        n_targets = target_theta_agent.shape[1]
        reg_loss_theta = self.loss_reg2(x_input_list=input_theta_tilde_agent[:, 0],
                                        x_target=target_theta_agent[:, 0])
        for target in range(1, n_targets):
            reg_loss_theta += self.loss_reg2(x_input_list=input_theta_tilde_agent[:, target],
                                             x_target=target_theta_agent[:, target])
        reg_loss_t = self.loss_reg1(x_input_list=input_t, target_id=0)
        for target_id in range(1, n_targets):
            reg_loss_t += self.loss_reg1(x_input_list=input_t, target_id=target_id)
        with open(os.path.join(LOG_PATH, 'contribution-loss-unlearn-1.log'), 'a') as f: # pylint: disable=unspecified-encoding
            f.write(
                f"{rec_loss_theta},{rec_loss_z},{reg_loss_theta/n_targets},{reg_loss_t/n_targets},"
                f"{rec_loss_theta + rec_loss_z + (reg_loss_theta + reg_loss_t)/n_targets}"
            )
            f.write('\n')
        return rec_loss_theta + rec_loss_z + (reg_loss_theta + reg_loss_t)/n_targets

class UnlearnLossUnnorm(nn.Module):
    """Blocks of loss functions
    Implemented:
        reconstruction loss: MSE loss with regularization l_rec(x1, x2) = ||x1 - x2||^2
        regularization loss: MSE cosine similarity loss
    """

    def __init__(self, sigmoid: bool = False) -> None:
        self.sigmoid = sigmoid
        super().__init__()

    def loss_rec(self, x_input, x_target):
        """
        x_target is the reference and x_input is the inferred input
        """
        if self.sigmoid:
            x_input = torch.sigmoid(x_input)
            x_target = torch.sigmoid(x_target)
        return F.mse_loss(input=x_input, target=x_target, reduction='sum')

    def loss_reg1(self, x_input_list, target_id):
        """Regularization loss."""
        n_agents = x_input_list.shape[1]
        ids = list(range(n_agents))
        ids.remove(target_id)
        cos = nn.CosineSimilarity(dim=1, eps=EPS)
        norm_coef = len(ids)
        x_target = x_input_list[:, target_id, :]
        cos_sum = 0
        angle_sum = 0
        if self.sigmoid:
            for i in ids:
                arg = x_input_list[:, i, :]
                cos_sum += F.mse_loss(
                    input=cos(
                        torch.sigmoid(x_target), torch.sigmoid(arg)
                    ) * torch.sigmoid(x_target),
                    target=torch.sigmoid(x_target),
                    reduction='sum'
                )
        else:
            for i in ids:
                arg = x_input_list[:, i, :]
                cos_score = cos(x_target, arg)
                cos_sum += F.mse_loss(input=cos_score * x_target, target=x_target, reduction='sum')
                angle_sum += cos_score
        return (-1/norm_coef) * cos_sum, angle_sum / norm_coef

    def loss_reg2(self, x_input_list, x_target):
        """Regularization loss."""
        cos = nn.CosineSimilarity(dim=1, eps=EPS)
        norm_coef = x_input_list.shape[1]
        cos_sum = 0
        angle_sum = 0
        if self.sigmoid:
            for i in range(norm_coef):
                arg = x_input_list[:, i, :]
                cos_sum += F.mse_loss(
                    input=cos(
                        torch.sigmoid(x_target), torch.sigmoid(arg)
                    ) * torch.sigmoid(x_target),
                    target=torch.sigmoid(x_target),
                    reduction='sum'
                )
        else:
            for i in range(norm_coef):
                arg = x_input_list[:, i, :]
                cos_score = cos(x_target, arg)
                cos_sum += F.mse_loss(
                    input=cos_score.view(-1, 1) * x_target,
                    target=x_target,
                    reduction='sum'
                )
                angle_sum += cos_score
        return (-1/norm_coef) * cos_sum, angle_sum / norm_coef

    def forward(
        self,
        input_theta: Tensor,
        target_theta: Tensor,
        input_z: Tensor,
        target_z: Tensor,
        input_t: Tensor,
        input_theta_tilde_agent: Tensor,
        target_theta_agent
    ) -> Tensor:
        """Forward method."""
        rec_loss_theta = self.loss_rec(x_input=input_theta, x_target=target_theta)
        rec_loss_z = self.loss_rec(x_input=input_z, x_target=target_z)

        n_targets = target_theta_agent.shape[1]

        reg_loss_theta, reg_theta_angle = self.loss_reg2(x_input_list=input_theta_tilde_agent[:, 0],
                                                         x_target=target_theta_agent[:, 0])
        for target in range(1, n_targets):
            tmp_theta, tmp_angle = self.loss_reg2(x_input_list=input_theta_tilde_agent[:, target],
                                                  x_target=target_theta_agent[:, target])
            reg_loss_theta += tmp_theta
            reg_theta_angle += tmp_angle

        reg_loss_t, reg_t_angle = self.loss_reg1(x_input_list=input_t, target_id=0)
        for target_id in range(1, n_targets):
            tmp_t, tmp_angle = self.loss_reg1(x_input_list=input_t, target_id=target_id)
            reg_loss_t += tmp_t
            reg_t_angle += tmp_angle

        with open(os.path.join(LOG_PATH, 'contribution-loss-unlearn-1.log'), 'a') as f: # pylint: disable=unspecified-encoding
            f.write(
                f"{rec_loss_theta},{rec_loss_z},{reg_loss_theta/n_targets},{reg_loss_t/n_targets},"
                f"{reg_theta_angle.item()/n_targets},{reg_t_angle.item()/n_targets},"
                f"{rec_loss_theta + rec_loss_z + (reg_loss_theta + reg_loss_t)/n_targets}"
            )
            f.write('\n')
        return rec_loss_theta + rec_loss_z + (reg_loss_theta + reg_loss_t) / n_targets
