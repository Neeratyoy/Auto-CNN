import numpy
import time
import argparse
import logging
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
from main_small import train
import torch
from BOHB_plotAnalysis import generateLossComparison, generateViz

class MyWorker(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval

    def compute(self, config, budget, **kwargs):
        # logging.info(config)
        # loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss,
        #              'mse': torch.nn.MSELoss}
        opti_dict = {'adam': torch.optim.Adam,
                     'adad': torch.optim.Adadelta,
                     'sgd': torch.optim.SGD}
                     # https://pytorch.org/docs/stable/optim.html
        # opti_aux_dict = {'adam': 'amsgrad', 'sgd': 'momentum', 'adad': None}

        try:
            test = kwargs.pop('test')
        except:
            test = False
        try:
            save = kwargs.pop('save')
        except:
            save = None
        logging.info("Tentative budget: "+str(budget*20/100))
        # dataset = 'KMNIST'
        # global dataset;
        data_dir = '../data'
        num_epochs = int(budget)
        batch_size = int(config['batch_size']) #50
        learning_rate = config['learning_rate']
        training_loss = torch.nn.CrossEntropyLoss  #loss_dict[config['training_criterion']]
        # if config['training_criterion'] == 'MSELoss':
        #         training_loss = torch.nn.MSELossid2conf = result.get_id2config_mapping()
        # else:
        #     training_loss = torch.nn.CrossEntropyLoss
        model_optimizer = opti_dict[config['model_optimizer']]
        # if config['model_optimizer'] == 'adam':
        #     opti_aux_param = bool(config['amsgrad'])    # Converting to bool, val of AMSGrad param
        # elif config['model_optimizer'] == 'sgd':
        #     opti_aux_param = config['momentum']
        # else:
        #     opti_aux_param = None

        # opti_aux_param = config[opti_aux_dict[config['model_optimizer']]] # Momentum or AMSGrad
        # if type(opti_aux_param) is not float and type(opti_aux_param) is not None:
        #     opti_aux_param = bool(opti_aux_param)
        # M = config['M']
        # N = config['N']
        # K = config['K']500
        # n_conv_layer = config['conv_layer'] # M*N
        # n_layers = config['fc_layer'] + n_conv_layer # M*N + K
        # maxpool_dict = {'True': True, 'False': False}
        # try:
        #     maxpool = maxpool_dict[config['maxpool']]
        # except KeyError:
        #     maxpool = False
        # activation = config['activation']

        train_score, train_loss, test_score, test_loss, train_time, test_time, total_model_params = train(
            dataset=dataset,  # dataset to use
            model_config=config,
            # {  # model architecture
            #     'n_layers': n_layers, #2,
            #     'n_conv_layer': n_conv_layer, #1
            #     'activation': activation,
            #     'maxpool': maxpool
            # },
            data_dir=data_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            train_criterion=training_loss,
            model_optimizer=model_optimizer,
            # opti_aux_param=opti_aux_param,
            data_augmentations=None,  # Not set in this example
            save_model_str=save,
            test=test
        )

        # res = numpy.clip(config['x'] + numpy.random.randn() / budget, config['x'] / 2, 1.5 * config['x'])
        # time.sleep(self.sleep_interval)

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
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        #########################
        # OPTIMIZER HYPERPARAMS #
        #########################
        alpha = CSH.UniformFloatHyperparameter('learning_rate', lower=0.00001, upper=0.1, default_value=0.001, log=True)
        opti = CSH.CategoricalHyperparameter('model_optimizer', choices=['adam', 'adad', 'sgd'], default_value='sgd')
        # amsgrad = CSH.CategoricalHyperparameter('amsgrad', choices=['True', 'False'], default_value='False')
        # ^ https://openreview.net/forum?id=ryQu7f-RZ
        # sgdmom = CSH.UniformFloatHyperparameter('momentum', lower=0, upper=0.99, default_value=0.90)
        # ^ https://distill.pub/2017/momentum/
        config_space.add_hyperparameters([alpha, opti])
        ###########################
        # OPTIMIZER CONDITIONALS  #
        ###########################
        # amsgrad_cond = CS.EqualsCondition(amsgrad, opti, 'adam')
        # sgdmom_cond = CS.EqualsCondition(sgdmom, opti, 'sgd')
        # config_space.add_conditions([amsgrad_cond, sgdmom_cond])
        ########################
        # TRAINING HYPERPARAMS #
        ########################
        # loss = CSH.CategoricalHyperparameter('training_criterion', choices=['cross_entropy'], default_value='cross_entropy') # choices=['mse', 'cross_entropy']
        # batch = CSH.UniformIntegerHyperparameter('batch_size', lower=32, upper=1024, default_value=128)
        batch = CSH.CategoricalHyperparameter('batch_size', choices=['50', '100', '200', '500', '1000'], default_value='100')
        # ^ https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
        # ^ https://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent
        config_space.add_hyperparameters([batch]) #, loss

        ############################
        # ARCHITECTURE HYPERPARAMS #
        ############################
        # n_conv_layer = CSH.UniformIntegerHyperparameter('n_conv_layer', lower=1, upper=3, default_value=1, log=False)
        # n_fc_layer = CSH.UniformIntegerHyperparameter('n_fc_layer', lower=1, upper=3, default_value=1, log=False)
        dropout = CSH.CategoricalHyperparameter('dropout', choices=['True', 'False'], default_value='False')
        activation = CSH.CategoricalHyperparameter('activation', choices=['relu', 'tanh', 'sigmoid'], default_value='tanh')
        batchnorm = CSH.CategoricalHyperparameter('batchnorm', choices=['True', 'False'], default_value='False')
        config_space.add_hyperparameters([dropout, activation, batchnorm]) # n_conv_layer, n_fc_layer,
        # LAYER 1 PARAMS
        kernel = CSH.CategoricalHyperparameter('kernel', choices=['3', '5', '7'], default_value='5')
        channel = CSH.UniformIntegerHyperparameter('channel', lower=3, upper = 12, default_value=3)
        padding = CSH.UniformIntegerHyperparameter('padding', lower=0, upper=3, default_value=2)
        stride = CSH.UniformIntegerHyperparameter('stride', lower=1, upper=2, default_value=1)
        maxpool = CSH.CategoricalHyperparameter('maxpool', choices=['True', 'False'], default_value='True')
        maxpool_kernel = CSH.UniformIntegerHyperparameter('maxpool_kernel', lower=2, upper=6, default_value=6)
        config_space.add_hyperparameters([kernel, padding, stride, channel, maxpool, maxpool_kernel])
        # LAYER 1 CONDITIONALS
        maxpool_cond = CS.NotEqualsCondition(maxpool, stride, 2)   # Convolution with stride 2 is equivalent to Maxpool
        maxpool_kernel_cond = CS.EqualsCondition(maxpool_kernel, maxpool, 'True')
        config_space.add_conditions([maxpool_cond, maxpool_kernel_cond])
        # LAYER 1 - RESTRICTING PADDING RANGE
        # Ensuring a padding domain of {0, 1, ..., floor(n/2)} for kernel_size n
        # padding_cond_0 = CS.ForbiddenAndConjunction(
        #         CS.ForbiddenEqualsClause(kernel, '3'),
        #         CS.ForbiddenInClause(padding, [2,3])
        # )
        # padding_cond_1 = CS.ForbiddenAndConjunction(
        #         CS.ForbiddenEqualsClause(kernel, '5'),
        #         CS.ForbiddenEqualsClause(padding, 3)
        # )
        # config_space.add_forbidden_clauses([padding_cond_0, padding_cond_1])

        # INTERMEDIATE FULLY CONNECTED LAYER PARAMS AND CONDITIONS (NOT OUTPUT LAYER)
        # Choosing min size as height/width of image, max as height x width of image
        # Max of 784 >>> output of the convolutions and pooling, hence adequately expressed
        # fc1 = CSH.UniformIntegerHyperparameter('fc_1', lower=28, upper=784, default_value=500, log=True)
        # # fc_dropout_1 = CSH.UniformFloatHyperparameter('fc_dropout_1', lower=0, upper=0.7, default_value=0)
        # fc2 = CSH.UniformIntegerHyperparameter('fc_2', lower=28, upper=784, default_value=500, log=True)
        # # fc_dropout_2 = CSH.UniformFloatHyperparameter('fc_dropout_2', lower=0, upper=0.7, default_value=0)
        # config_space.add_hyperparameters([fc1, fc2])
        # fc1_cond = CS.InCondition(fc1, n_fc_layer, [2,3])
        # fc2_cond = CS.EqualsCondition(fc2, n_fc_layer, 3)
        # config_space.add_conditions([fc1_cond, fc2_cond])

        return(config_space)


def testWorker(budget, test=False, save=None):
    worker = MyWorker(run_id = 'cnn_bohb')
    config = worker.get_configspace().sample_configuration()
    print(config)
    res = worker.compute(config = config, budget = budget, test=test, save=save)
    return(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", dest="dataset", type=str, default='KMNIST')
    parser.add_argument("--n_workers", dest="n_workers", type=int, default=1)
    parser.add_argument("--n_iterations", dest="n_iterations", type=int, default=1)
    parser.add_argument("--min_budget", dest="min_budget", type=int, default=1)
    parser.add_argument("--max_budget", dest="max_budget", type=int, default=2)
    parser.add_argument("--eta", dest="eta", type=int, default=2)
    parser.add_argument("--out_dir", dest="out_dir", type=str, default='bohb/')
    parser.add_argument("--run_id", dest="run_id", type=str, default='cnn_bohb')
    parser.add_argument("--show_plots", dest="show_plots", type=bool, default=False)
    args, kwargs = parser.parse_known_args()

    # print(args)
    # print(kwargs)
    global dataset
    dataset = args.dataset # Find way to pass to BOHB call sans config

    # Starting server to communicate between target algorithm and BOHB
    NS = hpns.NameServer(run_id=args.run_id, host='127.0.0.1', port=None)
    NS.start()
    w = MyWorker(sleep_interval = 0, nameserver='127.0.0.1', run_id=args.run_id)
    w.run(background=True)
    # Logging BOHB runs
    result_logger = hpres.json_result_logger(directory=args.out_dir, overwrite=True)
    # Configuring BOHB
    bohb = BOHB(  configspace = w.get_configspace(),
                  run_id = args.run_id, nameserver='127.0.0.1',
                  min_budget=args.min_budget, max_budget=args.max_budget,
                  eta = args.eta,
                  result_logger=result_logger,
                  min_points_in_model=13,
                  num_samples=13,
                  random_fraction=0.1,
                  top_n_percent=15)
    res = bohb.run(n_iterations=args.n_iterations )
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    # Waiting for all workers and services to shutdown
    time.sleep(2)

    # Extracting results
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    # Printing results
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/20))
    print('===' * 40)
    print('Best found configuration:', id2config[incumbent]['config'])
    print(res.get_runs_by_id(incumbent))
    print('===' * 40)
    print('~+~' * 40)
    # print("Generating plots for BOHB run")
    print('~+~' * 40)
    # try:
    #     generateLossComparison(args.out_dir, show = args.show_plots)
    #     generateViz(args.out_dir, show = args.show_plots)
    # except:
    #     print("Issue with plot generation! Not all plots may have been generated.")
    print('~+~' * 40)
    print('~+~' * 40)
    print('~+~' * 40)
    print('===' * 40)
    print("BUILDING AND EVALUATING INCUMBENT CONFIGURATION ON FULL TRAINING AND TEST SETS")
    print('===' * 40)
    result = hpres.logged_results_to_HBS_result(args.out_dir)
    id2conf = result.get_id2config_mapping()
    inc_id = result.get_incumbent_id()
    inc_config = id2conf[inc_id]['config']
    w = MyWorker('evaluating')
    res = w.compute(config=inc_config, budget=args.max_budget, test=True, save=None) #args.out_dir)
    print('~+~' * 40)
    # print("Training Accuracy: ", res['info']['train_score'])
    # print("Test Accuracy: ", res['info']['test_score'])
    print("Training Loss: ", res['info']['train_loss'])
    print("Test Loss: ", res['info']['test_score'])
