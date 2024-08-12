"""Experiment using FMNIST data."""
import os
import numpy as np

import ltu
from ltu.src_exp.exp_pipeline_classification import ExpPipelineClassification as ExpPipeline
from ltu.neural_nets.nets import FFN
from ltu.model.encoder_decoder import EncoderDecoder
from ltu.data import fmnist


LTU_PATH = ltu.__path__[0][0:-3]


class ExpFmnist:
    """Experiment FMNIST"""
    n_agents = 5
    samples = 500
    in_dim = 784
    data_path = fmnist.__path__[0] + '/'
    exp_name = 'fmnist'
    tasks = ['fmnist']
    n_rounds = 20
    agent_ids = [str(_) for _ in range(n_agents)]
    agent_ids_r = []
    unlearn_agents = [1]
    unlearn = False
    class_out = 10
    for i in range(n_agents):
        if i not in unlearn_agents:
            agent_ids_r.append(str(i))
    n_agents_r = n_agents - len(unlearn_agents)
    config = {'test_ratio': 0.9,
              'data_path': data_path,
              'tasks': tasks,
              'temp_path': os.path.join(LTU_PATH, 'temp'),
              'save_dir': os.path.join(LTU_PATH, f'results/{exp_name}/n_agents_{n_agents}/'),
              'use_best_model': False,
              'val_ratio': None,
              'batch_size': 50,
              'learning_rate': 1e-3}
    config_ll = {'num_epochs': 200, 'agent_ids': agent_ids}
    config_fl = {'num_epochs': 200, 'n_rounds': n_rounds, 'agent_ids': agent_ids}
    config_fl_r = {'num_epochs': 200, 'n_rounds': n_rounds, 'agent_ids': agent_ids_r}
    config_cl = {'num_epochs': 200, 'agent_ids': agent_ids}
    config_z = {
        'num_epochs': 1, 'learning_rate': 1e-3, 'memory_size': 1, 'if_print': True,
        'print_freq': .5, 'unlearn_agents': unlearn_agents, 'unlearn': True
    }
    config_z_r = {
        'num_epochs': 1, 'learning_rate': 1e-3, 'memory_size': 1, 'if_print': True,
        'print_freq': .5, 'unlearn_agents': unlearn_agents, 'unlearn': False
    }
    config_flz = {
        'num_epochs': 200, 'n_rounds': n_rounds, 'agent_ids': agent_ids, 'config_z': config_z
    }
    config_flz_r = {
        'num_epochs': 200, 'n_rounds': n_rounds, 'agent_ids': agent_ids_r, 'config_z': config_z_r
    }
    net = FFN(
        in_dim=in_dim, out_dim=10, hidden_dim=[], activation=[], drop_out=0.,
        seed=np.random.randint(0, 1e6, 1)
    )
    net_z = EncoderDecoder(
        num_agent=n_agents, input_dim=(n_agents * in_dim * class_out) + n_agents * class_out,
        z_dim=n_agents * 20, hidden_dim_t=20, hidden_dim_enc=in_dim, hidden_dim_dec=in_dim,
        class_out=class_out
    )
    net_z_r = EncoderDecoder(
        num_agent=n_agents_r, input_dim=(n_agents_r * in_dim) + n_agents_r,
        z_dim=n_agents_r * 20, hidden_dim_t=20, hidden_dim_enc=in_dim, hidden_dim_dec=in_dim,
        class_out=class_out
    )
    exp = ExpPipeline(config=config, exp_name='fl_ll')
    print('*** LL ***')
    exp.local_training(net, config_ll, save_name='ll')
    print('*** CL ***')
    exp.central_training(net, config_cl, save_name='cl')
    print('*** FL ***')
    exp.federated_training(net, config_fl, save_name='fl')
    print('*** FLZ ***')
    exp.federated_unlearning_training(net, net_z, config_flz, save_name='flz')


if __name__ == '__main__':
    ef = ExpFmnist()
