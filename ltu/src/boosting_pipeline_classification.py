"""Classification Pipeline."""
from copy import deepcopy
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as MyDataNormalizer
from sklearn.model_selection import train_test_split
import torch

from .read_data import SimpleRead
from .agent_model import AgentModel, CentralModel
from .federated_averaging import FederatedAveraging, IsolatedTraining
from ..tools.performance_evaluator import ClassificationEvaluator as MyClassificationEvaluator
from ..tools.convergence_evaluator import NumberOfRounds


EPSILON = 1e-20


class FedAvePipeLine(object):
    """Federated Averaging pipeline."""
    def __init__(
        self,
        agent_ids: list,
        data_obj,
        test_ratio: float = 0.9,
        val_ratio: float = 0.1,
    ):
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.agent_ids = agent_ids
        self.sr = data_obj
        self.data_dict = {}
        samples_train = []
        samples_test = []
        for target_agent in agent_ids:
            data_dict = self.sr.read(agent_id=target_agent)
            if isinstance(data_dict['dataset']['X'], pd.DataFrame):
                x = data_dict['dataset']['X'].to_numpy()
                y = data_dict['dataset']['Y'][self.sr.tasks].to_numpy()
            else:
                x = data_dict['dataset']['X']
                y = data_dict['dataset']['Y'].reshape(-1, 1)
            x_train_target, x_test_target, y_train_target, y_test_target = \
                train_test_split(x, y, test_size=self.test_ratio, shuffle=False)

            self.data_dict.update({target_agent: {'x_train': x_train_target,
                                                  'x_test': x_test_target,
                                                  'y_train': y_train_target,
                                                  'y_test': y_test_target}
                                   })
            samples_train.append([sum(y_train_target[:, 0] == _) for _ in range(10)])
            samples_test.append([sum(y_test_target[:, 0] == _) for _ in range(10)])
            self.data_dict.update({'samples_train': np.array(samples_train).T,
                                   'samples_test': np.array(samples_test).T})


class AgentLocalTraining(FedAvePipeLine):
    """Agent local training module."""
    def __init__(
        self,
        local_trainer,
        target_agent: str,
        agent_ids: list,
        data_obj,
        test_ratio: float,
        val_ratio: float
    ):
        super().__init__(agent_ids, data_obj, test_ratio, val_ratio)
        self.target_agent = target_agent
        self.am = AgentModel(federated_scheme=IsolatedTraining(),
                             data_normalizer=MyDataNormalizer(),
                             local_trainer=deepcopy(local_trainer),
                             evaluator=MyClassificationEvaluator(tasks=self.sr.tasks),
                             validation_ratio=self.val_ratio)
        self._target_agent_local_trainer = None
        self._eval_dict = None

    @property
    def target_agent_local_trainer(self):
        """Local trainer of the target agent."""
        return self._target_agent_local_trainer

    @property
    def eval_dict(self):
        """Evaluation results."""
        return self._eval_dict

    def fit_predict_evaluate(self):
        """Fit, predict and evaluate."""
        target_data = self.data_dict[self.target_agent]
        self.am.fit_predict_evaluate(
            x_train=target_data['x_train'],
            y_train=target_data['y_train'],
            x_test=target_data['x_test'],
            y_test=target_data['y_test']
        )
        print(f'local training agent: {self.target_agent}, {self.am.eval_dict}')
        self._eval_dict = self.am.eval_dict
        self._target_agent_local_trainer = self.am.local_trainer


class AgentCentralTraining(FedAvePipeLine):
    """Central training."""
    def __init__(
        self,
        local_trainer,
        agent_ids: list,
        data_obj: SimpleRead,
        test_ratio: float,
        val_ratio: float
    ):
        super().__init__(agent_ids, data_obj, test_ratio, val_ratio)
        x_train_all = self.data_dict[self.agent_ids[0]]['x_train']
        y_train_all = self.data_dict[self.agent_ids[0]]['y_train']
        for agent in self.agent_ids[1:]:
            x_train_agent = self.data_dict[agent]['x_train']
            y_train_agent = self.data_dict[agent]['y_train']
            x_train_all = np.vstack([x_train_all, x_train_agent])
            y_train_all = np.vstack([y_train_all, y_train_agent])
        self.cm = CentralModel(federated_scheme=IsolatedTraining(),
                               data_normalizer=MyDataNormalizer(),
                               local_trainer=deepcopy(local_trainer),
                               evaluator=MyClassificationEvaluator(self.sr.tasks),
                               validation_ratio=self.val_ratio)
        self._x_train_all = x_train_all
        self._y_train_all = y_train_all
        self._central_trainer = None
        self._eval_dict = {}

    @property
    def eval_dict(self):
        """Evaluation results."""
        return self._eval_dict

    @property
    def central_trainer(self):
        """Central trainer."""
        return self._central_trainer

    def fit_predict_evaluate(self):
        """Fit, predict and evaluate."""
        self.cm.fit(x=self._x_train_all, y=self._y_train_all)
        self._central_trainer = deepcopy(self.cm.local_trainer)
        self._x_train_all = None
        self._y_train_all = None
        for target_agent in self.agent_ids:
            self.cm.predict(x=self.data_dict[target_agent]['x_test'])
            self.cm.evaluate(y=self.data_dict[target_agent]['y_test'])
            print(f'central training agent: {target_agent}, {self.cm.eval_dict}')
            self._eval_dict.update({target_agent: self.cm.eval_dict})


class AgentMultiRoundFederatedAveraging(FedAvePipeLine):
    """Multi-round federated averaging."""
    def __init__(
        self,
        local_trainer,
        n_rounds: int,
        agent_ids: list,
        data_obj: SimpleRead,
        test_ratio: float,
        val_ratio: float,
        if_print: bool = True
    ):
        super().__init__(agent_ids, data_obj, test_ratio, val_ratio)
        self.if_print = if_print
        self._global_trainer = deepcopy(local_trainer)
        self._federated_trainer = None
        self._eval_dict_list = []
        self.fa = FederatedAveraging(convergence_evaluator=NumberOfRounds(n_rounds=n_rounds))

    @property
    def eval_dict_list(self):
        """Evaluation results."""
        return self._eval_dict_list

    @property
    def federated_trainer(self):
        """Federated trainer."""
        return self._federated_trainer

    def fit_predict_evaluate(self):
        """Fit, predict and evaluate."""
        federated_trainer = deepcopy(self._global_trainer)
        agent_model_dict = {}
        r = 0
        while self.fa.convergence_evaluator.terminate(r) is False:
            if self.if_print is True:
                print(f'round {r}')
            eval_dict = {}
            for name in self.agent_ids:
                am = AgentModel(
                    federated_scheme=self.fa,
                    data_normalizer=MyDataNormalizer(),
                    local_trainer=deepcopy(federated_trainer),
                    evaluator=MyClassificationEvaluator(self.sr.tasks),
                    validation_ratio=self.val_ratio
                )
                am.fit_predict_evaluate(
                    x_train=self.data_dict[name]['x_train'],
                    y_train=self.data_dict[name]['y_train'],
                    x_test=self.data_dict[name]['x_test'],
                    y_test=self.data_dict[name]['y_test']
                )
                if self.if_print is True:
                    print(f'federated training agent: {name}, {am.eval_dict}')
                agent_model_dict.update({name: am.local_trainer})
                eval_dict.update({name: am.eval_dict})
            self._eval_dict_list.append(eval_dict)
            federated_trainer = self.fa.average(agent_model_dict=agent_model_dict)
            r += 1
        self._federated_trainer = deepcopy(federated_trainer)


class AgentMultiRoundFederatedAveragingZ(FedAvePipeLine):
    """Multi-round federated averaging with weighted averaging."""
    def __init__(
        self,
        local_trainer,
        contribution_trainer,
        memory_size: int,
        n_rounds: int,
        agent_ids: list,
        data_obj: SimpleRead,
        test_ratio: float,
        val_ratio: float,
        if_print: bool = True
    ):
        super().__init__(agent_ids, data_obj, test_ratio, val_ratio)
        self.if_print = if_print
        self._global_trainer = local_trainer
        self._federated_trainer = None
        self._eval_dict_list = []
        self.fa = FederatedAveraging(convergence_evaluator=NumberOfRounds(n_rounds=n_rounds))
        self.contribution_trainer = contribution_trainer
        self.memory_size = memory_size
        self._theta_memory = None

    @property
    def eval_dict_list(self):
        """Evaluation dict."""
        return self._eval_dict_list

    @property
    def federated_trainer(self):
        """Federated trainer."""
        return self._federated_trainer

    def fit_predict_evaluate(
        self,
        unlearn_agents_id=None,
        unlearn=False
    ):
        """Fit, predict, and evaluate."""
        if unlearn_agents_id is None:
            unlearn_agents_id = []
        federated_trainer = deepcopy(self._global_trainer)
        agent_model_dict = {}
        r = 0
        while self.fa.convergence_evaluator.terminate(r) is False:
            if self.if_print is True:
                print(f'round {r}')
            eval_dict = {}
            for name in self.agent_ids:
                am = AgentModel(
                    federated_scheme=self.fa,
                    data_normalizer=MyDataNormalizer(),
                    local_trainer=deepcopy(federated_trainer),
                    evaluator=MyClassificationEvaluator(self.sr.tasks),
                    validation_ratio=self.val_ratio
                )
                am.fit_predict_evaluate(
                    x_train=self.data_dict[name]['x_train'],
                    y_train=self.data_dict[name]['y_train'],
                    x_test=self.data_dict[name]['x_test'],
                    y_test=self.data_dict[name]['y_test']
                )
                if self.if_print is True:
                    print(f'federated training agent: {name}, {am.eval_dict}')
                agent_model_dict.update({name: am.local_trainer})
                eval_dict.update({name: am.eval_dict})
            self._eval_dict_list.append(eval_dict)
            theta_dict = self._get_agent_theta_dict(agent_model_dict)
            theta = torch.cat(
                [theta_dict[agent] for agent in theta_dict.keys()], 1  # pylint: disable=C0206, C0201
            )
            if self._theta_memory is None:
                self._theta_memory = theta.repeat(self.memory_size, 1)
            else:
                self._theta_memory = torch.vstack((self._theta_memory[1:, :], theta))

            n_agents = len(theta_dict.keys())
            self.contribution_trainer.fit(theta=self._theta_memory, n_agents=n_agents)
            agent_z = self.contribution_trainer.get_agent_z()

            agent_theta_hat = self.contribution_trainer.get_agent_theta_hat()
            federated_trainer = self.fa.average_z(
                agent_theta_hat=agent_theta_hat,
                agent_z=agent_z,
                ref_model=agent_model_dict,
                class_out=self.contribution_trainer.model.class_out
            )
            r += 1
        self._federated_trainer = deepcopy(federated_trainer)

        if len(unlearn_agents_id) > 0 and unlearn:
            agent_theta_hat = self.contribution_trainer.get_agent_theta_hat()
            federated_trainer = self.fa.average_z_unlearn(
                agent_theta_hat=agent_theta_hat,
                agent_z=agent_z,
                ref_model=agent_model_dict,
                unlearn_agents_id=unlearn_agents_id,
                class_out=self.contribution_trainer.model.class_out
            )

            eval_dict = {}
            for name in self.agent_ids:
                if eval(name) not in unlearn_agents_id:  # pylint: disable=W0123
                    am = AgentModel(federated_scheme=self.fa,
                                    data_normalizer=MyDataNormalizer(),
                                    local_trainer=deepcopy(federated_trainer),
                                    evaluator=MyClassificationEvaluator(self.sr.tasks),
                                    validation_ratio=self.val_ratio)

                    am.predict_evaluate(x_train=self.data_dict[name]['x_train'],
                                        y_train=self.data_dict[name]['y_train'],
                                        x_test=self.data_dict[name]['x_test'],
                                        y_test=self.data_dict[name]['y_test'])
                    eval_dict.update({name: am.eval_dict})
            self._eval_dict_list.append(eval_dict)
            self._federated_trainer = deepcopy(federated_trainer)

    @staticmethod
    def _get_agent_theta_dict(agent_model_dict) -> Dict:
        #  TODO: this needs to be generalized.  # pylint: disable=W0511
        theta_dict = {}
        for agent in agent_model_dict.keys():
            weight = agent_model_dict[agent].best_model.state_dict()['net_input.0.weight']
            bias = agent_model_dict[agent].best_model.state_dict()['net_input.0.bias'].view(-1, 1)
            theta_dict.update({agent: torch.cat([weight, bias], 1).view(1, -1)})
        return theta_dict
