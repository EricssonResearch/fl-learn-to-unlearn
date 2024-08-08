"""Experiment pipeline."""

import os
import pickle
from copy import deepcopy
import numpy as np

import ltu
from ..src.boosting_pipeline_classification import (
    SimpleRead,
    AgentLocalTraining,
    AgentCentralTraining,
    AgentMultiRoundFederatedAveragingZ,
    AgentMultiRoundFederatedAveraging
)
from ..src.models_classifier import MLPClassifier
from ..src.contribution_model import ContributionModel

EPSILON = 1e-20


LOG_PATH = os.path.join(ltu.__path__[0][0:-3], 'log')
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)


class ExpPipelineClassification:
    """Classification pipeline."""
    def __init__(self, config, exp_name):
        self.exp_name = exp_name
        self.config = config
        self._test_ratio = config['test_ratio']
        self._data_path = config['data_path']
        self._tasks = config['tasks']
        self._temp_path = config['temp_path']
        self._save_dir = (
            config['save_dir'] + str(config['test_ratio']) + '_' + str(config['val_ratio']) + '/'
        )

    def local_training(self, init_net, config, save_name):
        """Local training."""
        for target_agent in config['agent_ids']:
            init_net.update()
            target_net = deepcopy(init_net)
            alt = AgentLocalTraining(
                local_trainer=MLPClassifier(
                    model=target_net,
                    num_epochs=config['num_epochs'],
                    learning_rate=self.config['learning_rate'],
                    batch_size=self.config['batch_size'],
                    use_best_model=self.config['use_best_model'],
                    save_path=self._temp_path,
                    min_num_epochs=1,
                    if_print=False
                ),
                target_agent=target_agent,
                agent_ids=config['agent_ids'],
                data_obj=SimpleRead(data_path=self._data_path, tasks=self._tasks),
                test_ratio=self._test_ratio,
                val_ratio=self.config['val_ratio']
            )
            alt.fit_predict_evaluate()
            self.save(
                alt.eval_dict,
                save_dir=self._save_dir+'agent_'+target_agent+'/'+self.exp_name+'/',
                name=save_name
            )

        print('\n-------------------------------------------')

    def central_training(self, init_net, config, save_name):
        """Central training."""
        act = AgentCentralTraining(
            local_trainer=MLPClassifier(
                model=deepcopy(init_net),
                num_epochs=config['num_epochs'],
                learning_rate=self.config['learning_rate'],
                batch_size=self.config['batch_size'],
                use_best_model=self.config['use_best_model'],
                min_num_epochs=1,
                save_path=self._temp_path,
                if_print=False
            ),
            agent_ids=config['agent_ids'],
            data_obj=SimpleRead(data_path=self._data_path, tasks=self._tasks),
            test_ratio=self._test_ratio,
            val_ratio=self.config['val_ratio']
        )
        act.fit_predict_evaluate()
        ave_kpi = [
            np.mean(
                [act.eval_dict[key][task]['f1-score'] for key in config['agent_ids']]
            ) for task in self._tasks
        ]
        print(f'cl: ave_kpi={ave_kpi}')
        for target_agent in config['agent_ids']:
            self.save(act.eval_dict[target_agent],
                      save_dir=self._save_dir + 'agent_' + target_agent + '/' + self.exp_name + '/',
                      name=save_name)

    @staticmethod
    def save(result, save_dir, name):
        """Save model."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_dir + name + '__kpi.pickle', 'wb') as handle:
            pickle.dump(result, handle)

    def federated_unlearning_training(self, init_net, init_z_net, config, save_name):
        """Federated unlearning."""
        amf = AgentMultiRoundFederatedAveragingZ(
            local_trainer=MLPClassifier(
                model=deepcopy(init_net),
                num_epochs=config['num_epochs'],
                learning_rate=self.config['learning_rate'],
                batch_size=self.config['batch_size'],
                use_best_model=self.config[
                'use_best_model'],
                min_num_epochs=1,
                save_path=self._temp_path,
                if_print=False
            ),
            contribution_trainer=ContributionModel(
                model=init_z_net,
                num_epochs=config['config_z']['num_epochs'],
                learning_rate=config['config_z']['learning_rate'],
                if_print=config['config_z']['if_print'],
                print_freq=config['config_z']['print_freq']
            ),
            memory_size=config['config_z']['memory_size'],
            n_rounds=config['n_rounds'],
            agent_ids=config['agent_ids'],
            data_obj=SimpleRead(data_path=self._data_path, tasks=self._tasks),
            test_ratio=self._test_ratio,
            val_ratio=self.config['val_ratio'],
            if_print=True
        )
        amf.fit_predict_evaluate(
            config['config_z']['unlearn_agents'], config['config_z']['unlearn']
        )
        ave_kpi = [np.mean([amf.eval_dict_list[-2][key][task]['f1-score'] for
                            key in config['agent_ids']]) for task in self._tasks]
        with open(os.path.join(LOG_PATH, 'exp_incl-_subset.log'), 'a') as f:  # pylint: disable=W1514
            f.write(f' flz: ave_kpi = {ave_kpi}')
            f.write('\t')
        ave_kpi_unlearn = [
            np.mean(
                [
                    amf.eval_dict_list[-1][key][task]['f1-score'] for key in
                    config['agent_ids'] if eval(key) not in config['config_z']['unlearn_agents'] # pylint: disable=W0123
                ]
            ) for task in self._tasks
        ]
        with open(os.path.join(LOG_PATH, 'exp_incl-_subset.log'), 'a') as f:  # pylint: disable=W1514
            f.write(f' flz: ave_kpi_unlearn = {ave_kpi_unlearn}')
            f.write('\n')
        print(f' flz: ave_kpi_unlearn = {ave_kpi_unlearn}')

        for target_agent in config['agent_ids']:
            if eval(target_agent) not in config['config_z']['unlearn_agents']:  # pylint: disable=W0123
                self.save(
                    [amf.eval_dict_list[_][target_agent] for _ in range(len(amf.eval_dict_list))],
                    save_dir=self._save_dir + 'agent_' + target_agent + '/' + self.exp_name + '/',
                    name=save_name
                )

    def federated_training(self, init_net, config, save_name):
        """Federated training."""
        amf = AgentMultiRoundFederatedAveraging(
            local_trainer=MLPClassifier(
                model=deepcopy(init_net),
                num_epochs=config['num_epochs'],
                learning_rate=self.config['learning_rate'],
                batch_size=self.config['batch_size'],
                use_best_model=
                self.config['use_best_model'],
                min_num_epochs=1,
                save_path=self._temp_path,
                if_print=False
            ),
            n_rounds=config['n_rounds'],
            agent_ids=config['agent_ids'],
            data_obj=SimpleRead(data_path=self._data_path, tasks=self._tasks),
            test_ratio=self._test_ratio,
            val_ratio=self.config['val_ratio'],
            if_print=True
        )
        amf.fit_predict_evaluate()
        ave_kpi = [np.mean([amf.eval_dict_list[-1][key][task]['f1-score'] for
                            key in config['agent_ids']]) for task in self._tasks]
        print(f' fl: ave_kpi = {ave_kpi}')

        with open(os.path.join(LOG_PATH, 'exp_incl-_subset.log'), 'a') as f:  # pylint: disable=W1514
            f.write(f' fl: ave_kpi = {ave_kpi}')
            f.write('\n')
        for target_agent in config['agent_ids']:
            self.save(
                [amf.eval_dict_list[_][target_agent] for _ in range(len(amf.eval_dict_list))],
                save_dir=self._save_dir + 'agent_' + target_agent + '/' + self.exp_name + '/',
                name=save_name
            )
