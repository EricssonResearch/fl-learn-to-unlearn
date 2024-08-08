"""Encoder-Decoder Models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    """Encoder-Decoder model.
        Args:
        - num_agent: number of agents.
        - input_dim: dimension of concatenated model parameters (thetas).
        - z_dim: dimension of latent contribution variables z.
        - hidden_dim_t: dimension of latent variable from the inner encoder-decoder t.
        - hidden_dim_enc: hidden dimension within the encoder.
        - hidden_dim_dec: hidden dimension within the decoder.
        - class_out: number of output classes.

    """

    def __init__(
        self,
        num_agent: int,
        input_dim: int,
        z_dim: int,
        hidden_dim_t: int,
        hidden_dim_enc: int,
        hidden_dim_dec: int,
        class_out: int,
    ):
        super(EncoderDecoder, self).__init__()
        assert z_dim % num_agent == 0, "z_dim should be a multiple of num_agent"
        assert input_dim % num_agent == 0, "input_dim should be a multiple of num_agent"
        self.class_out = class_out
        self.z_dim = z_dim
        self.num_agent = num_agent
        self.phi_enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_enc),
            nn.ReLU(),
            nn.Linear(hidden_dim_enc, num_agent * z_dim),
            nn.ReLU()
        )
        self.psi_encs = []
        for _ in range(num_agent):
            self.psi_encs.append(nn.Linear(z_dim, hidden_dim_t))
        self.psi_decs = []
        for _ in range(num_agent):
            self.psi_decs.append(nn.Linear(hidden_dim_t, z_dim // num_agent))
        self.phi_dec = nn.Sequential(
            nn.Linear(num_agent * z_dim, hidden_dim_dec),
            nn.ReLU(),
            nn.Linear(hidden_dim_dec, input_dim),
            nn.ReLU()
        )

    def _project(self, s, z_hat, thetas_hat):
        """
        projects z_hat to thetas space with s.
        """
        dim = thetas_hat.shape[-1]
        thetas_hat_tilde = torch.zeros(
            (thetas_hat.shape[0], self.num_agent, self.num_agent - 1, dim//self.num_agent)
        )
        thetas_hat = thetas_hat.reshape(thetas_hat.shape[0], self.num_agent, dim//self.num_agent)
        for a in range(self.num_agent):
            for b in range(self.num_agent - 1):
                thetas_hat_tilde[:, a, b, :] = torch.mul(
                    z_hat[:, a, :], s[:, a, b, :]
                ).sum(dim=1)[:, None] * thetas_hat[:, a, :]

        return thetas_hat_tilde

    def forward(self, thetas):
        """
        The forward function of the encoder-decoder.
        Args:
            thetas: input model params from all agents.
        Returns:
            - thetas_hat: R^input_dim
            - thetas_hat_tilde: R^{num_agent * (num_agent - 1) * (input_dim / num_agent)}
            - z: R^{num_agent * (z_dim / num_agent)}
            - z_hat: R^{num_agent * (z_dim / num_agent)}
            - s: R^{num_agent * (num_agent - 1) * (input_dim / num_agent)}
        """

        n_samples = thetas.shape[0]
        agent_dim = self.z_dim // self.num_agent
        x = self.phi_enc(thetas)
        z = x[:, 0:self.z_dim]
        s_vec = x[:, self.z_dim:]
        s = s_vec.reshape(n_samples, self.num_agent, self.num_agent-1, agent_dim)
        t = []
        for i in range(self.num_agent):
            t.append(F.relu(self.psi_encs[i](z)))
        z_hat = []
        for i in range(self.num_agent):
            z_hat.append(F.relu(self.psi_decs[i](t[i])))
        z_hat_vec = torch.stack(z_hat, 1)
        thetas_hat = self.phi_dec(torch.cat((z_hat_vec.view(n_samples, -1), s_vec), 1))
        thetas_hat_tilde = self._project(s, z_hat_vec, thetas_hat)
        t_dim = t[0].shape[-1]
        t = torch.cat(t, 1)
        t = t.reshape(n_samples, self.num_agent, t_dim)
        return thetas_hat, thetas_hat_tilde, z, z_hat_vec.view(n_samples, -1), t
