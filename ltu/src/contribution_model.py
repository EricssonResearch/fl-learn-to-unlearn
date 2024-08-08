"""Implementation of the contribution model."""

import torch

from ..loss.unlearning_loss import UnlearnLossUnnorm


class ContributionModel:
    """Contribution model."""
    def __init__(
        self,
        model,
        num_epochs,
        learning_rate,
        if_print=False,
        print_freq=.1
    ):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.if_print = if_print
        self.print_freq = print_freq
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = UnlearnLossUnnorm(sigmoid=False)
        self._z_hat = None
        self._theta_hat = None
        self._n_agents = None

    def fit(self, theta, n_agents):
        """Fit method."""
        self._n_agents = n_agents
        n_samples, dim = theta.shape
        for epoch in range(self.num_epochs):
            self._theta_hat, thetas_hat_tilde, z, self._z_hat, t = self.model.forward(theta)
            theta_hat_agent = self._theta_hat.reshape(
                n_samples, self._n_agents, dim//self._n_agents
            )
            train_loss = self.criterion(
                input_theta=self._theta_hat,
                target_theta=theta,
                input_z=self._z_hat,
                target_z=z,
                input_t=t,
                input_theta_tilde_agent=thetas_hat_tilde,
                target_theta_agent=theta_hat_agent
            )
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            if (self.if_print is True) and (epoch % (self.num_epochs * self.print_freq) == 0):
                print(
                    'Train:      epoch [{}/{}], loss:{:.4f}'.format(  # pylint: disable=C0209
                        epoch + 1, self.num_epochs, train_loss.item()
                    )
                )

        self.get_agent_z()
        self.get_agent_theta_hat()

    def get_agent_z(self):
        """Get latent embeddings for the agent."""
        memory_size = self._z_hat.shape[0]
        dim = self._z_hat.shape[1]
        return self._z_hat.reshape(memory_size, self._n_agents, dim//self._n_agents)[-1, :, :]

    def get_agent_theta_hat(self):
        """Get the theta_hat for the agent."""
        memory_size = self._theta_hat.shape[0]
        dim = self._theta_hat.shape[1]
        return self._theta_hat.reshape(memory_size, self._n_agents, dim//self._n_agents)[-1, :, :]
