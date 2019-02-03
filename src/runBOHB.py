import numpy
import time
import logging
import argparse
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
from BOHBvisualization import generateViz  # script
from main import train
import torch

class MyWorker(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval

    def compute(self, config, budget, **kwargs):
        """
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        print("\n\nBudget: ", int(budget), "\n\n")
        tr_acc, tr_loss, val_acc, val_loss, tr_time, val_time, params = train(
            dataset='KMNIST',  # dataset to use
            model_config = { 'n_layers': 2, 'n_conv_layer': 1 },
            data_dir='../data',
            num_epochs=int(budget),
            batch_size=50,
            learning_rate=config['learning_rate'],
            # train_criterion=torch.nn.MSELoss,
            train_criterion=torch.nn.CrossEntropyLoss,
            model_optimizer=torch.optim.Adam,
            data_augmentations=None,  # Not set in this example
            save_model_str=None
        )
        # Call train
        # Handle train-validation
        logging.info("Absolutely ready to return!")
        return ({
            'loss': float(val_loss),  # this is the a mandatory field to run hyperband
            'info': {'train_score':tr_acc, 'train_loss':tr_loss, 'validation_score':val_acc, 'validation_loss':val_loss,
                     'train_time':tr_time, 'test_time':val_time, 'total_model_params': params}  # can be used for any user-defined information - also mandatory
        })


def get_configspace():
    config_space = CS.ConfigurationSpace()
    alpha = CSH.UniformFloatHyperparameter('learning_rate', lower=0.0001, upper=0.1, default_value=0.001)
    config_space.add_hyperparameter(alpha)
    return(config_space)


def runBOHB(iter, n_workers, visualize=True):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n_workers", dest="n_workers", type=int, default=n_workers)
    parser.add_argument("--n_iterations", dest="n_iterations", type=int, default=iter)
    parser.add_argument("--min_budget", dest="min_budget", type=int, default=2)
    parser.add_argument("--max_budget", dest="max_budget", type=int, default=3)
    parser.add_argument("--eta", dest="eta", type=int, default=3)
    parser.add_argument("--out_dir", dest="out_dir", type=str, default='bohb/toy_runs')
    args, kwargs = parser.parse_known_args()
    run_id = 'cnn_bohb'
    # Start a Nameserver
    NS = hpns.NameServer(run_id=run_id, host='127.0.0.1', port=None)
    ns_host, ns_port = NS.start()
    # Start a worker
    worker = MyWorker(sleep_interval = 0, nameserver='127.0.0.1',run_id=run_id)
    workers = []
    for i in range(args.n_workers):
        print("Start worker %d" % i)
        worker = MyWorker(nameserver=ns_host, nameserver_port=ns_port, run_id=run_id, id=i)
        worker.run(background=True)
        workers.append(worker)
    # Log outputs
    result_logger = hpres.json_result_logger(directory=args.out_dir, overwrite=True)
    # Run an Optimizer
    bohb = BOHB(  configspace = get_configspace(),
                  run_id = run_id, nameserver='127.0.0.1',
                  min_budget=args.min_budget, max_budget=args.max_budget,
                  result_logger=result_logger)
    res = bohb.run(n_iterations=args.n_iterations)
    print("\n\nFinished BOHB runs\n\n")
    # Stop all services
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    print("Shutting down BOHB services\n\n")
    # Analysis of the results
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/args.max_budget))

    # visualizing outputs of BOHB
    # if visualize:
    #     generateViz(args.out_dir)


def testWorker(budget):
    worker = MyWorker(run_id = 'cnn_bohb')
    config = get_configspace().sample_configuration()
    print(config)
    res = worker.compute(config = config, budget = budget)
    return(res)


if __name__== '__main__':
    runBOHB(1, 2, True)