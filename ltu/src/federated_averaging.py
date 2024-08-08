"""Script performing federated averaging."""

from copy import deepcopy
import torch

EPSILON = 1e-16


class IsolatedTraining:
    """Isolated """
    name = 'isolated training'

    @staticmethod
    def learn_local(x_train, y_train, x_val, y_val, trainer):
        """Local training."""
        trainer.fit(x_train, y_train, x_val, y_val)
        return trainer

    @staticmethod
    def predict_local(x, trainer):
        """Local prediction."""
        return trainer.predict(x)

    @staticmethod
    def evaluate(y, y_hat, evaluator):
        """Evaluation of the model."""
        return evaluator.evaluate(y, y_hat)


class FederatedAveraging(IsolatedTraining):
    """Federated averaging."""
    name = 'federated averaging'

    def __init__(self, convergence_evaluator):
        self.convergence_evaluator = convergence_evaluator

    @staticmethod
    def average(agent_model_dict):
        """Simple averaging as in standard FL."""
        agent_names = list(agent_model_dict.keys())
        n_agents = len(agent_names)
        layers = list(agent_model_dict[agent_names[0]].best_model.state_dict().keys())

        average_model = deepcopy(agent_model_dict[agent_names[0]])
        state_dict_ave = deepcopy(agent_model_dict[agent_names[0]].best_model.state_dict())
        for layer in layers:
            state_dict_layer_ave = agent_model_dict[
                agent_names[0]].best_model.state_dict()[layer] / n_agents
            for agent in agent_names[1:]:
                state_dict_layer_ave += (
                    agent_model_dict[agent].best_model.state_dict()[layer] / n_agents)
            state_dict_ave.update({layer: state_dict_layer_ave})

        average_model.best_model.state_dict().update(state_dict_ave)
        average_model.best_model.load_state_dict(state_dict_ave)
        average_model.model.state_dict().update(state_dict_ave)
        average_model.model.load_state_dict(state_dict_ave)
        return average_model

    @staticmethod
    def average_z(agent_theta_hat, agent_z, ref_model, class_out=1):
        """Weighted averaging which replaces the standard averaging.
        """
        agent_names = list(ref_model.keys())
        average_model = deepcopy(ref_model[agent_names[0]])
        pi_agents = torch.sum(agent_z, 1)
        pi = torch.sum(pi_agents)
        theta_bar = torch.sum(agent_theta_hat * (pi_agents/pi).view(-1, 1), 0)
        print(f'contributions per agent: {(pi_agents/pi).detach()}')

        #  TODO: we need to generalize this  # pylint: disable=W0511
        state_dict_ave = deepcopy(average_model.best_model.state_dict())
        len_w = state_dict_ave['net_input.0.weight'].shape[1]

        state_dict_ave['net_input.0.weight'] = theta_bar[0:len_w * class_out].view(class_out, -1)
        state_dict_ave['net_input.0.bias'] = theta_bar[len_w * class_out:]
        average_model.best_model.state_dict().update(state_dict_ave)
        average_model.best_model.load_state_dict(state_dict_ave)
        average_model.model.state_dict().update(state_dict_ave)
        average_model.model.load_state_dict(state_dict_ave)
        return average_model

    @staticmethod
    def average_z_unlearn(
        agent_theta_hat, agent_z, ref_model, unlearn_agents_id: list, class_out=1):
        """Federated averaging in LTU."""
        agent_names = list(ref_model.keys())
        agent_names_int = [eval(x) for x in agent_names]  # pylint: disable=W0123
        assert set(unlearn_agents_id).issubset(set(agent_names_int)), \
            "agent(s) to be unlearned must be a subset of all agents participated."

        average_model = deepcopy(ref_model[agent_names[0]])

        retain_ids = [x for x in agent_names_int if x not in unlearn_agents_id]

        pi_agents = torch.sum(agent_z[retain_ids, :], 1)
        pi = torch.sum(pi_agents)
        theta_bar = torch.sum(agent_theta_hat[retain_ids, :] * (pi_agents / pi).view(-1, 1), 0)

        msg = 'contributions per agent after unlearning from agents'
        print(
            f'{msg} {unlearn_agents_id}: {(pi_agents/pi).detach()}'
        )

        # TODO: we need to generalize this  # pylint: disable=W0511
        state_dict_ave = deepcopy(average_model.best_model.state_dict())
        len_w = state_dict_ave['net_input.0.weight'].shape[1]

        state_dict_ave['net_input.0.weight'] = theta_bar[0:len_w * class_out].view(class_out, -1)
        state_dict_ave['net_input.0.bias'] = theta_bar[len_w * class_out:]
        average_model.best_model.state_dict().update(state_dict_ave)
        average_model.best_model.load_state_dict(state_dict_ave)
        average_model.model.state_dict().update(state_dict_ave)
        average_model.model.load_state_dict(state_dict_ave)

        return average_model
