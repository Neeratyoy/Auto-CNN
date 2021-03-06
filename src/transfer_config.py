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
    '''
    The Worker class to run BOHB
    '''
    # The class variable that is required for initialising the object - the parent configuration
    old_config = None

    def __init__(self, *args, old_config=None, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)
        # Parent configuration
        self.old_config = old_config
        # Remnant of the BOHB example that was tested and incrementally developed
        # upon. Leaving it here as a relic I suppose.
        self.sleep_interval = sleep_interval

    def compute(self, config, budget, **kwargs):
        '''
        The primary function of the worker class which is called by BOHB at each iteration to call the
        Target Algorithm for a given configutation and budget
        :param config: The configuration passed by BOHB to the Target Algorithm as a JSON from class ConfigSpace
        :param budget: The numeric budget that BOHB allots for a particular run with the given config
        :param kwargs: Optional parameters that can be passed
        :return: A dict containing 'loss' (signal to BOHB which it minimizes) and 'info' (containing additional info)
        '''
        loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss,
                     'mse': torch.nn.MSELoss}
        opti_dict = {'adam': torch.optim.Adam,
                     'adad': torch.optim.Adadelta,
                     'sgd': torch.optim.SGD}
                     # https://pytorch.org/docs/stable/optim.html
        opti_aux_dict = {'adam': 'amsgrad', 'sgd': 'momentum', 'adad': None}

        old_config = self.old_config

        # Additional parameters that the Target Algorithm may need other than config and budget
        try:
            # Evaluates the Test set or splits the Training to evaluate on Validation
            test = kwargs.pop('test')
        except:
            test = False
        try:
            # To save the model or not
            save = kwargs.pop('save')
        except:
            save = None

        # All source code for this project lies in src/ while the data dump is in data/
        data_dir = '../data'
        # For a neural net training, the alloted budget == # of epochs
        num_epochs = int(budget)
        batch_size = int(config['batch_size'])
        learning_rate = old_config['learning_rate']
        # Fixing the loss to CrossEntropy and not hyperparameterizing the loss
        training_loss = torch.nn.CrossEntropyLoss
        # Checking for the type of optimizer and looking for the relevant auxiliary parameter
        model_optimizer = opti_dict[old_config['model_optimizer']]
        if old_config['model_optimizer'] == 'adam':
            if old_config['amsgrad'] == 'True':
                opti_aux_param = True
            else:
                opti_aux_param = False
        elif old_config['model_optimizer'] == 'sgd':
            opti_aux_param = old_config['momentum']
        else:
            opti_aux_param = None
        data_augmentation = None # config['aug_prob'] # Not None when used

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
            data_augmentations=data_augmentation,
            save_model_str=save,
            test=test,
            old_config=old_config
        )
        # test_loss == Validation/Test depending on the previously defined test=True/False parameter
        return ({
            'loss': float(test_loss),  # this is the a mandatory field to run hyperband
            'info': {'train_score':float(train_score), 'train_loss':float(train_loss),
                    'test_score':float(test_score), 'test_loss':float(test_loss),
                     'train_time':float(train_time), 'test_time':float(test_time),
                     'total_model_params': float(total_model_params)}  # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace(reference_config):
        '''
        Defines the configuration space for the Target Algorithm - the CNN module in this case
        :param reference_config: The Parent configuration/incumbent from the first BOHB run
        :return: a ConfigSpace object containing the hyperparameters, conditionals and forbidden clauses on them
        '''
        config_space = CS.ConfigurationSpace()
        batch = CSH.UniformIntegerHyperparameter('batch_size', lower=100, upper=1000, default_value=100, log=True)
        # ^ https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
        # ^ https://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent
        # aug_prob = CSH.UniformFloatHyperparameter('aug_prob', lower=0, upper=0.5, default_value=0)
        config_space.add_hyperparameters([batch])

        ############################
        # ARCHITECTURE HYPERPARAMS #
        ############################
        n_fc_layer = CSH.UniformIntegerHyperparameter('n_fc_layer', lower=1, upper=3, default_value=1, log=False)
        fc_nodes = CSH.UniformIntegerHyperparameter('fc_nodes', lower=50, upper=784, default_value=500, log=True)
        config_space.add_hyperparameters([n_fc_layer, fc_nodes])

        n_layers = reference_config['n_conv_layer']
        if n_layers >= 1:
            channel_1 = CSH.UniformIntegerHyperparameter('channel_1', lower=reference_config['channel_1'], upper=20,
                                                         default_value=reference_config['channel_1'])
            config_space.add_hyperparameter(channel_1)
        if n_layers >= 2:
            channel_2 = CSH.UniformIntegerHyperparameter('channel_2', lower=1, upper=4, default_value=2)
            config_space.add_hyperparameter(channel_2)
        if n_layers == 3:
            if reference_config['kernel_3'] == '1':
                channel_3 = CSH.UniformFloatHyperparameter('channel_3', lower=0.5, upper=1, default_value=0.5)
            else:
                channel_3 = CSH.UniformIntegerHyperparameter('channel_3', lower=1, upper=3, default_value=2)
            config_space.add_hyperparameter(channel_3)

        fc_cond = CS.InCondition(fc_nodes, n_fc_layer, [2,3])
        config_space.add_condition(fc_cond)

        return(config_space)


def save_config(source, dest, name):
    '''
    Reads the incumbent from a BOHB output directory and writes it as a JSON in the specified directory
    :param source: Directory from where to read the incumbent
    :param dest: Directory to save the file
    :param name: Name given to the JSON being saved
    :return: void
    '''
    result = hpres.logged_results_to_HBS_result(source)
    id2conf = result.get_id2config_mapping()
    inc_id = result.get_incumbent_id()
    inc_config = id2conf[inc_id]['config']
    f = open(dest+name+'.json', 'w')
    f.write(json.dumps(inc_config))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', "--dataset", dest="dataset", type=str, default='KMNIST', choices=['KMNIST', 'K49'],
                        help='Which dataset to evaluate on {K49, KMNIST}')
    # parser.add_argument("--n_workers", dest="n_workers", type=int, default=1)
    parser.add_argument('-i', "--n_iterations", dest="n_iterations", type=int, default=1,
                        help='Number of BOHB iterations that will be run')
    parser.add_argument('-b', "--min_budget", dest="min_budget", type=int, default=1,
                        help='Minimum budget alloted to BOHB')
    parser.add_argument('-B', "--max_budget", dest="max_budget", type=int, default=2,
                        help='Maximum budget alloted to BOHB (>min_budget)')
    parser.add_argument('-e', "--eta", dest="eta", type=int, default=2,
                        help='The fraction of configurations passed to next budget by BOHB')
    parser.add_argument('-o', "--out_dir", dest="out_dir", type=str, default='bohb/',
                        help='The output directory to write all results (Will be overwritten)')
    parser.add_argument('-c', "--config_dir", dest="config_dir", type=str,
                        help='Directory that has config.json and results.json from a BOHB run')
    parser.add_argument('-r', "--run_id", dest="run_id", type=str, default='cnn_bohb',
                        help='Any specific run ID for annotation')
    parser.add_argument('-s', "--show_plots", dest="show_plots", type=bool, choices=[True, False], default=False,
                        help='To decide if plots additionally need to be opened in additional windows')
    parser.add_argument('-v', '--verbose', default='INFO', choices=['INFO', 'DEBUG'], help='verbosity')
    args, kwargs = parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    start_time = time.time()

    # Reading parent configuration
    result = hpres.logged_results_to_HBS_result(args.config_dir)
    id2conf = result.get_id2config_mapping()
    inc_id = result.get_incumbent_id()
    inc_config = id2conf[inc_id]['config']

    global dataset
    dataset = args.dataset # Find way to pass to BOHB call sans config

    # Starting server to communicate between target algorithm and BOHB
    NS = hpns.NameServer(run_id=args.run_id, host='127.0.0.1', port=None)
    NS.start()
    # Initialising the worker class (only one worker)
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
                  random_fraction=0.1,
                  num_samples=4)
    res = bohb.run(n_iterations=args.n_iterations )
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    # Waiting for all workers and services to shutdown
    print('='*40)
    print("Time taken for BOHB: ", time.time() - start_time)
    print('='*40)
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
    print(res.get_runs_by_id(incumbent)[-1]['info'])
    # Saving incumbent configuration as a JSON - not necessary
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
