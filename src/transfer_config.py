import numpy
import time
import argparse
import logging
import json
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
from main_transfer_config import train
import torch
from BOHB_plotAnalysis import generateLossComparison, generateViz

class TransferWorker(Worker):
    old_config = None

    def __init__(self, *args, old_config=None, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.old_config = old_config

        self.sleep_interval = sleep_interval

    def compute(self, config, budget, **kwargs):
        # logging.info(config)
        loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss,
                     'mse': torch.nn.MSELoss}
        opti_dict = {'adam': torch.optim.Adam,
                     'adad': torch.optim.Adadelta,
                     'sgd': torch.optim.SGD}
                     # https://pytorch.org/docs/stable/optim.html
        opti_aux_dict = {'adam': 'amsgrad', 'sgd': 'momentum', 'adad': None}

        old_config = self.old_config # kwargs.pop('old_config')

        try:
            test = kwargs.pop('test')
        except:
            test = False
        try:
            save = kwargs.pop('save')
        except:
            save = None

        # dataset = 'KMNIST'
        # global dataset;
        data_dir = '../data'
        num_epochs = int(budget)
        batch_size = int(config['batch_size']) #50
        learning_rate = old_config['learning_rate']
        training_loss = torch.nn.CrossEntropyLoss # loss_dict[old_config['training_criterion']]
        # if config['training_criterion'] == 'MSELoss':
        #         training_loss = torch.nn.MSELossid2conf = result.get_id2config_mapping()
        # else:
        #     training_loss = torch.nn.CrossEntropyLoss
        model_optimizer = opti_dict[old_config['model_optimizer']]
        if old_config['model_optimizer'] == 'adam':
            if old_config['amsgrad'] == 'True':
                opti_aux_param = True
            else:
                opti_aux_param = False
        elif load_config['model_optimizer'] == 'sgd':
            opti_aux_param = load_config['momentum']
        else:
            opti_aux_param = None

        train_score, train_loss, test_score, test_loss, train_time, test_time, total_model_params, _ = train(
            dataset=dataset,  # dataset to use
            model_config=config,
            data_dir=data_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            train_criterion=training_loss,
            model_optimizer=model_optimizer,
            opti_aux_param=opti_aux_param,
            data_augmentations=None,  # Not set in this example
            save_model_str=save,
            test=test,
            old_config=old_config
        )

        return ({
            'loss': float(test_loss),  # this is the a mandatory field to run hyperband
            # 'loss': float(1-test_score),  # this is the a mandatory field to run hyperband
            # 'loss': float(res),  # this is the a mandatory field to run hyperband
            'info': {'train_score':float(train_score), 'train_loss':float(train_loss),
                    'test_score':float(test_score), 'test_loss':float(test_loss),
                     'train_time':float(train_time), 'test_time':float(test_time),
                     'total_model_params': float(total_model_params)}  # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace(reference_config):
        config_space = CS.ConfigurationSpace()
        # batch = CSH.CategoricalHyperparameter('batch_size', choices=['100', '200', '500', '1000'], default_value='100')
        batch = CSH.UniformIntegerHyperparameter('batch_size', lower=100, upper=1000, default_value=100, log=True)
        # ^ https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
        # ^ https://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent

        ############################
        # ARCHITECTURE HYPERPARAMS #
        ############################
        n_fc_layer = CSH.UniformIntegerHyperparameter('n_fc_layer', lower=1, upper=3, default_value=1, log=False)
        fc_nodes = CSH.UniformIntegerHyperparameter('fc_nodes', lower=50, upper=784, default_value=500, log=True)
        # dropout = CSH.CategoricalHyperparameter('dropout', choices=['True', 'False'], default_value='False')
        # batchnorm = CSH.CategoricalHyperparameter('batchnorm', choices=['True', 'False'], default_value='False')
        config_space.add_hyperparameters([n_fc_layer, batch, fc_nodes])

        n_layers = reference_config['n_conv_layer']
        if n_layers >= 1:
            channel_1 = CSH.UniformIntegerHyperparameter('channel_1', lower=reference_config['channel_1'], upper=24, default_value=reference_config['channel_1'])
            config_space.add_hyperparameter(channel_1)
        if n_layers >= 2:
            channel_2 = CSH.UniformIntegerHyperparameter('channel_2', lower=2, upper=4, default_value=2)
            config_space.add_hyperparameter(channel_2)
        if n_layers == 3:
            if reference_config['kernel_3'] == '1':
                # channel_3 = CSH.CategoricalHyperparameter('channel_3', choices=['0.5'], default_value='0.5')
                channel_3 = CSH.UniformFloatHyperparameter('channel_3', lower=0.5, upper=1, default_value=0.5)
            else:
                channel_3 = CSH.UniformIntegerHyperparameter('channel_3', lower=1, upper=3, default_value=2)
                # channel_3 = CSH.CategoricalHyperparameter('channel_3', choices=['0.5', '1', '2', '3'], default_value='2')
            config_space.add_hyperparameter(channel_3)

        fc_cond = CS.InCondition(fc_nodes, n_fc_layer, [2,3])
        config_space.add_condition(fc_cond)

        return(config_space)

def save_config(source, dest, name):
    result = hpres.logged_results_to_HBS_result(source)
    id2conf = result.get_id2config_mapping()
    inc_id = result.get_incumbent_id()
    inc_config = id2conf[inc_id]['config']
    f = open(dest+name+'.json', 'w')
    f.write(json.dumps(inc_config))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", dest="dataset", type=str, default='KMNIST')
    parser.add_argument("--n_workers", dest="n_workers", type=int, default=1)
    parser.add_argument("--n_iterations", dest="n_iterations", type=int, default=1)
    parser.add_argument("--min_budget", dest="min_budget", type=int, default=1)
    parser.add_argument("--max_budget", dest="max_budget", type=int, default=2)
    parser.add_argument("--eta", dest="eta", type=int, default=3)
    parser.add_argument("--out_dir", dest="out_dir", type=str, default='bohb/')
    parser.add_argument("--config_dir", dest="config_dir", type=str, default='bohb/KMNIST/4_5_15/')
    parser.add_argument("--run_id", dest="run_id", type=str, default='cnn_bohb')
    parser.add_argument("--show_plots", dest="show_plots", type=bool, default=False)
    args, kwargs = parser.parse_known_args()

    start_time = time.time()

    result = hpres.logged_results_to_HBS_result(args.config_dir)
    id2conf = result.get_id2config_mapping()
    inc_id = result.get_incumbent_id()
    inc_config = id2conf[inc_id]['config']

    # print(args)
    # print(kwargs)
    global dataset
    dataset = args.dataset # Find way to pass to BOHB call sans config

    # Starting server to communicate between target algorithm and BOHB
    NS = hpns.NameServer(run_id=args.run_id, host='127.0.0.1', port=None)
    NS.start()
    w = TransferWorker(old_config=inc_config, nameserver='127.0.0.1', run_id=args.run_id)
    w.run(background=True)
    # Logging BOHB runs
    result_logger = hpres.json_result_logger(directory=args.out_dir, overwrite=True)
    # Configuring BOHB
    bohb = BOHB(  configspace = w.get_configspace(inc_config),
                  run_id = args.run_id, nameserver='127.0.0.1',
                  min_budget=args.min_budget, max_budget=args.max_budget,
                  eta = args.eta,
                  result_logger=result_logger,
                  # min_points_in_model=7,
                  random_fraction=0.1,
                  top_n_percent=15)
    res = bohb.run(n_iterations=args.n_iterations )
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    # Waiting for all workers and services to shutdown
    end_time = time.time()
    time.sleep(2)
    # Saving parent config - IMPORTANT for final numbers
    save_config(args.config_dir, args.out_dir, 'parent')

    # Extracting results
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    # Printing results
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/20))
    print('===' * 40)
    print('Best found configuration:', id2config[incumbent]['config'])
    print(id2config[incumbent]['info'])
    save_config(args.out_dir, args.out_dir, 'best')

    print('===' * 40)
    print('~+~' * 40)
    print("Generating plots for BOHB run")
    print('~+~' * 40)
    try:
        generateLossComparison(args.out_dir, show = args.show_plots)
        generateViz(args.out_dir, show = args.show_plots)
    except:
        print("Issue with plot generation! Not all plots may have been generated.")
    print('~+~' * 40)

    print("Time taken for BOHB: ", end_time - start_time)
