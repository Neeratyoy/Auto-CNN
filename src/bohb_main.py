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
from main import train
import torch
from BOHB_plotAnalysis import generateLossComparison, generateViz

class MyWorker(Worker):
    '''
    The Worker class to run BOHB# logging.info(config)
    '''
    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)
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
        # Dictionaries that map aliases defined in the configuration space to Python (or PyTorch) code
        loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss,
                     'mse': torch.nn.MSELoss}
        opti_dict = {'adam': torch.optim.Adam,
                     'adad': torch.optim.Adadelta,
                     'sgd': torch.optim.SGD}
                     # https://pytorch.org/docs/stable/optim.html
        opti_aux_dict = {'adam': 'amsgrad', 'sgd': 'momentum', 'adad': None}

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
        learning_rate = config['learning_rate']
        # Fixing the loss to CrossEntropy and not hyperparameterizing the loss
        training_loss = torch.nn.CrossEntropyLoss
        # Checking for the type of optimizer and looking for the relevant auxiliary parameter
        model_optimizer = opti_dict[config['model_optimizer']]
        if config['model_optimizer'] == 'adam':
            if config['amsgrad'] == 'True':
                opti_aux_param = True
            else:
                opti_aux_param = False
        elif config['model_optimizer'] == 'sgd':
            opti_aux_param = config['momentum']
        else:
            opti_aux_param = None

        data_augmentation = None #config['aug_prob']

        # Call to the target algorithm
        train_score, train_loss, test_score, test_loss, train_time, test_time, total_model_params, _, _ = train(
            dataset=dataset,  # dataset to use
            model_config=config,
            data_dir=data_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            train_criterion=training_loss,
            model_optimizer=model_optimizer,
            opti_aux_param=opti_aux_param,
            data_augmentations=data_augmentation,  # Not set in this example
            save_model_str=save,
            test=test
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
    def get_configspace():
        '''
        Defines the configuration space for the Target Algorithm - the CNN module in this case
        :return: a ConfigSpace object containing the hyperparameters, conditionals and forbidden clauses on them
        '''
        config_space = CS.ConfigurationSpace()
        #########################
        # OPTIMIZER HYPERPARAMS #
        #########################
        alpha = CSH.UniformFloatHyperparameter('learning_rate', lower=0.00001, upper=0.1, default_value=0.001, log=True)
        opti = CSH.CategoricalHyperparameter('model_optimizer', choices=['adam', 'adad', 'sgd'], default_value='sgd')
        amsgrad = CSH.CategoricalHyperparameter('amsgrad', choices=['True', 'False'], default_value='False')
        # ^ https://openreview.net/forum?id=ryQu7f-RZ
        sgdmom = CSH.UniformFloatHyperparameter('momentum', lower=0, upper=0.99, default_value=0.90)
        # ^ https://distill.pub/2017/momentum/
        config_space.add_hyperparameters([alpha, opti, amsgrad, sgdmom])
        ###########################
        # OPTIMIZER CONDITIONALS  #
        ###########################
        amsgrad_cond = CS.EqualsCondition(amsgrad, opti, 'adam')
        sgdmom_cond = CS.EqualsCondition(sgdmom, opti, 'sgd')
        config_space.add_conditions([amsgrad_cond, sgdmom_cond])

        ########################
        # TRAINING HYPERPARAMS #
        ########################
        # loss = CSH.CategoricalHyperparameter('training_criterion', choices=['cross_entropy'],
        #                                       default_value='cross_entropy')
        # aug_prob = CSH.UniformFloatHyperparameter('aug_prob', lower=0, upper=0.5, default_value=0)
        batch = CSH.CategoricalHyperparameter('batch_size', choices=['50', '100', '200', '500', '1000'],
                                               default_value='100')
        # ^ https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
        # ^ https://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent
        config_space.add_hyperparameters([batch])

        ############################
        # ARCHITECTURE HYPERPARAMS #
        ############################
        n_conv_layer = CSH.UniformIntegerHyperparameter('n_conv_layer', lower=1, upper=3, default_value=1, log=False)
        n_fc_layer = CSH.UniformIntegerHyperparameter('n_fc_layer', lower=1, upper=3, default_value=1, log=False)
        dropout = CSH.CategoricalHyperparameter('dropout', choices=['True', 'False'], default_value='False')
        activation = CSH.CategoricalHyperparameter('activation', choices=['relu', 'tanh', 'sigmoid'],
                                                   default_value='tanh')
        batchnorm = CSH.CategoricalHyperparameter('batchnorm', choices=['True', 'False'], default_value='False')
        config_space.add_hyperparameters([n_conv_layer, n_fc_layer, dropout, activation, batchnorm])
        #
        # LAYER 1 PARAMS
        #
        kernel_1 = CSH.CategoricalHyperparameter('kernel_1', choices=['3', '5', '7'], default_value='5')
        channel_1 = CSH.UniformIntegerHyperparameter('channel_1', lower=3, upper = 12, default_value=3)
        padding_1 = CSH.UniformIntegerHyperparameter('padding_1', lower=0, upper=3, default_value=2)
        stride_1 = CSH.UniformIntegerHyperparameter('stride_1', lower=1, upper=2, default_value=1)
        maxpool_1 = CSH.CategoricalHyperparameter('maxpool_1', choices=['True', 'False'], default_value='True')
        maxpool_kernel_1 = CSH.UniformIntegerHyperparameter('maxpool_kernel_1', lower=2, upper=6, default_value=6)
        config_space.add_hyperparameters([kernel_1, padding_1, stride_1, maxpool_1, maxpool_kernel_1, channel_1])
        # LAYER 1 CONDITIONALS
        maxpool_cond_1 = CS.NotEqualsCondition(maxpool_1, stride_1, 2)
        # ^ Convolution with stride 2 is equivalent to Maxpool - https://arxiv.org/abs/1412.6806
        maxpool_kernel_cond_1 = CS.EqualsCondition(maxpool_kernel_1, maxpool_1, 'True')
        config_space.add_conditions([maxpool_cond_1, maxpool_kernel_cond_1])
        # LAYER 1 - RESTRICTING PADDING RANGE
        # Ensuring a padding domain of {0, 1, ..., floor(n/2)} for kernel_size n
        padding_1_cond_0 = CS.ForbiddenAndConjunction(
                CS.ForbiddenEqualsClause(kernel_1, '3'),
                CS.ForbiddenInClause(padding_1, [2,3])
        )
        padding_1_cond_1 = CS.ForbiddenAndConjunction(
                CS.ForbiddenEqualsClause(kernel_1, '5'),
                CS.ForbiddenEqualsClause(padding_1, 3)
        )
        config_space.add_forbidden_clauses([padding_1_cond_0, padding_1_cond_1])

        #
        # LAYER 2 PARAMS
        #
        kernel_2 = CSH.CategoricalHyperparameter('kernel_2', choices=['3', '5', '7'], default_value='5')
        # Channels for Layer 2 onwards is a multiplicative factor of previous layer's channel size
        channel_2 = CSH.CategoricalHyperparameter('channel_2', choices=['1', '2', '3', '4'], default_value='2')
        # ^ Categorical instead of Integer owing to the design choice of channel_3 - for parity's sake I suppose
        padding_2 = CSH.UniformIntegerHyperparameter('padding_2', lower=0, upper=3, default_value=2)
        stride_2 = CSH.UniformIntegerHyperparameter('stride_2', lower=1, upper=2, default_value=1)
        maxpool_2 = CSH.CategoricalHyperparameter('maxpool_2', choices=['True', 'False'], default_value='True')
        maxpool_kernel_2 = CSH.UniformIntegerHyperparameter('maxpool_kernel_2', lower=2, upper=6, default_value=6)
        config_space.add_hyperparameters([kernel_2, padding_2, stride_2, maxpool_2, maxpool_kernel_2, channel_2])
        # LAYER 2 CONDITIONALS
        maxpool_cond_2 = CS.NotEqualsCondition(maxpool_2, stride_2, 2)
        # ^ Convolution with stride 2 is equivalent to Maxpool - https://arxiv.org/abs/1412.6806
        maxpool_kernel_cond_2 = CS.EqualsCondition(maxpool_kernel_2, maxpool_2, 'True')
        # LAYER 2 - RESTRICTING PADDING RANGE
        # Ensuring a padding domain of {0, 1, ..., floor(n/2)} for kernel_size n
        padding_2_cond_0 = CS.ForbiddenAndConjunction(
                CS.ForbiddenEqualsClause(kernel_2, '3'),
                CS.ForbiddenInClause(padding_2, [2,3])
        )
        padding_2_cond_1 = CS.ForbiddenAndConjunction(
                CS.ForbiddenEqualsClause(kernel_2, '5'),
                CS.ForbiddenEqualsClause(padding_2, 3)
        )
        config_space.add_forbidden_clauses([padding_2_cond_0, padding_2_cond_1])
        # LAYER 2 ACTIVATE CONDITION
        # Layer 2 params will activate optionally only if n_conv_layer >= 2
        kernel_2_cond = CS.InCondition(kernel_2, n_conv_layer, [2, 3])
        channel_2_cond = CS.InCondition(channel_2, n_conv_layer, [2, 3])
        padding_2_cond = CS.InCondition(padding_2, n_conv_layer, [2, 3])
        stride_2_cond = CS.InCondition(stride_2, n_conv_layer, [2, 3])
        maxpool_2_cond = CS.AndConjunction(CS.InCondition(maxpool_2, n_conv_layer, [2, 3]), maxpool_cond_2)
        maxpool_kernel_2_cond = CS.AndConjunction(CS.InCondition(maxpool_kernel_2, n_conv_layer, [2, 3]),
                                                  maxpool_kernel_cond_2)
        config_space.add_conditions([kernel_2_cond, channel_2_cond, padding_2_cond, stride_2_cond,
                                     maxpool_2_cond, maxpool_kernel_2_cond])

        #
        # LAYER 3 PARAMS
        #
        kernel_3 = CSH.CategoricalHyperparameter('kernel_3', choices=['1', '3', '5', '7'], default_value='5')
        # Channels for Layer 2 onwards is a multiplicative factor of previous layer's channel size
        # Also being the max convolution layer allowed, this allows for 1x1 convolution
        # Therefore, a downsampling of channel depth (factor of 0.5) - reduce dimensions along depth
        channel_3 = CSH.CategoricalHyperparameter('channel_3', choices=['0.5', '1', '2', '3'], default_value='2')
        padding_3 = CSH.UniformIntegerHyperparameter('padding_3', lower=0, upper=3, default_value=2)
        stride_3 = CSH.UniformIntegerHyperparameter('stride_3', lower=1, upper=2, default_value=1)
        maxpool_3 = CSH.CategoricalHyperparameter('maxpool_3', choices=['True', 'False'], default_value='True')
        maxpool_kernel_3 = CSH.UniformIntegerHyperparameter('maxpool_kernel_3', lower=2, upper=6, default_value=6)
        config_space.add_hyperparameters([kernel_3, padding_3, stride_3, maxpool_3, maxpool_kernel_3, channel_3])
        # LAYER 3 CONDITIONALS
        maxpool_cond_3 = CS.NotEqualsCondition(maxpool_3, stride_3, 2)
        maxpool_kernel_cond_3 = CS.EqualsCondition(maxpool_kernel_3, maxpool_3, 'True')
        # LAYER 3 - RESTRICTING PADDING RANGE
        # Ensuring a padding domain of {0, 1, ..., floor(n/2)} for kernel_size n
        padding_3_cond_0 = CS.ForbiddenAndConjunction(
                CS.ForbiddenEqualsClause(kernel_3, '3'),
                CS.ForbiddenInClause(padding_3, [2,3])
        )
        padding_3_cond_1 = CS.ForbiddenAndConjunction(
                CS.ForbiddenEqualsClause(kernel_3, '5'),
                CS.ForbiddenEqualsClause(padding_3, 3)
        )
        config_space.add_forbidden_clauses([padding_3_cond_0, padding_3_cond_1])
        # LAYER 3 ACTIVATE CONDITION
        # Layer 2 params will activate optionally only if n_conv_layer >= 3 (max 3 conv layers allowed currently)
        kernel_3_cond = CS.EqualsCondition(kernel_3, n_conv_layer, 3)
        channel_3_cond = CS.EqualsCondition(channel_3, n_conv_layer, 3)
        padding_3_cond = CS.EqualsCondition(padding_3, n_conv_layer, 3)
        stride_3_cond = CS.EqualsCondition(stride_3, n_conv_layer, 3)
        maxpool_3_cond = CS.AndConjunction(CS.InCondition(maxpool_3, n_conv_layer, [2, 3]), maxpool_cond_3)
        maxpool_kernel_3_cond = CS.AndConjunction(CS.InCondition(maxpool_kernel_3, n_conv_layer, [2, 3]), maxpool_kernel_cond_3)
        config_space.add_conditions([kernel_3_cond, channel_3_cond, padding_3_cond, stride_3_cond, maxpool_3_cond, maxpool_kernel_3_cond])

        # COMPLICATED ASSUMPTIONS MADE EMPIRICALLY TO IMPOSE CONSTRAINTS ON VARIOUS PARAMETERS SUCH THAT THE
        # CONFIGURATIONS SAMPLED BY THE CONFIGURATOR DOESN'T YIELD AN ARCHITECTURE WITH SHAPE/DIMENSION MISMATCH
        # FOLLOWING BASIC ASSUMPTIONS WERE MADE:
        #   1) AT MAX 3 CONVOLUTION LAYERS CAN BE FORMED
        #   2) CONVOLUTION KERNEL SIZE DOMAIN : {3, 5, 7}
        #   3) EACH CONVOLUTION LAYER MAY OR MAY NOT HAVE A MAXPOOL LAYER
        #   4) MAXPOOL KERNEL SIZE DOMAIN : {2, 3, 4, 5, 6}
        #   5) A CONVOLUTION WITH STRIDE 2 IS EQUIVALENT TO MAXPOOL - cannot occur together in same layer
        # MANY OTHER CONDITIONS WERE ADDED BASED ON OBSERVATION (a couple of them mentioned below):
        #   1) If n_conv_layer=3 then cannot have maxpool on all 3 layers
        #   2) Cannot use a convolution kernel of size 5 or 7 in the third layer
        #   ...
        for_two_layers_1 = CS.ForbiddenAndConjunction(
                # Disallowing large maxpool kernel in first layer for a 2-layer convoluiton
                CS.ForbiddenEqualsClause(n_conv_layer, 2),
                CS.ForbiddenInClause(maxpool_kernel_1, [3,4,5,6]),
                CS.ForbiddenInClause(maxpool_kernel_2, [4,5,6])
        )
        for_two_layers_2 = CS.ForbiddenAndConjunction(
                # Disallowing large convolution filter following a large max pool
                CS.ForbiddenInClause(maxpool_kernel_1, [5,6]),
                CS.ForbiddenInClause(kernel_2, ['5', '7'])
        )
        for_two_layers_3 = CS.ForbiddenAndConjunction(
                # Disallowing large convolution filter following a large max pool
                CS.ForbiddenInClause(kernel_1, ['5', '7']),
                CS.ForbiddenEqualsClause(maxpool_1, 'True'),
                CS.ForbiddenInClause(kernel_2, ['5', '7'])
        )
        for_three_layers_1_0 = CS.ForbiddenAndConjunction(
                # Constraining maxpool kernel sizes for a 3 layer convolution
                # Small maxpool kernel if subsequent layer contains another maxpool
                CS.ForbiddenEqualsClause(n_conv_layer, 3),
                CS.ForbiddenInClause(maxpool_kernel_1, [5,6])
        )
        for_three_layers_1_1 = CS.ForbiddenAndConjunction(
                # Constraining maxpool kernel sizes for a 3 layer convolution
                # Small maxpool kernel if subsequent layer contains another maxpool
                CS.ForbiddenEqualsClause(n_conv_layer, 3),
                CS.ForbiddenInClause(maxpool_kernel_2, [4,5,6])
        )
        for_three_layers_1_2 = CS.ForbiddenAndConjunction(
                # Constraining maxpool kernel sizes for a 3 layer convolution
                # Small maxpool kernel if subsequent layer contains another maxpool
                CS.ForbiddenEqualsClause(n_conv_layer, 3),
                CS.ForbiddenInClause(maxpool_kernel_3, [3, 4,5,6])
        )
        for_three_layers_2 = CS.ForbiddenAndConjunction(
                # Constraining maxpool kernel sizes for a 3 layer convolution
                # Small maxpool kernel if subsequent layer contains another maxpoo)l
                CS.ForbiddenEqualsClause(n_conv_layer, 3),
                CS.ForbiddenInClause(maxpool_kernel_1, [3,4,5,6]),
                CS.ForbiddenInClause(maxpool_kernel_3, [5,6])
        )
        for_three_layers_3 = CS.ForbiddenAndConjunction(
                # Constraining maxpool kernel sizes for a 3 layer convolution
                CS.ForbiddenEqualsClause(n_conv_layer, 3),
                CS.ForbiddenInClause(maxpool_kernel_2, [3,4,5,6]),
                CS.ForbiddenInClause(maxpool_kernel_3, [5,6])
        )
        for_three_layers_4 = CS.ForbiddenAndConjunction(
                # Constraining maxpool kernel sizes for a 3 layer convolution
                CS.ForbiddenEqualsClause(n_conv_layer, 3),
                CS.ForbiddenEqualsClause(stride_2, 2),
                CS.ForbiddenInClause(maxpool_kernel_3, [5,6])
        )
        for_three_layers_5 = CS.ForbiddenAndConjunction(
                # Disallowing large convolution filter following a large max pool
                CS.ForbiddenEqualsClause(n_conv_layer, 3),
                CS.ForbiddenEqualsClause(stride_1, 2),
                CS.ForbiddenEqualsClause(stride_2, 2),
                CS.ForbiddenInClause(maxpool_kernel_3, [3,4,5,6])
        )
        for_three_layers_6 = CS.ForbiddenAndConjunction(
                # Disallowing large convolution filter following a large max pool
                CS.ForbiddenInClause(maxpool_kernel_2, [4,5,6]),
                CS.ForbiddenInClause(kernel_3, ['5', '7'])
        )
        for_three_layers_7 = CS.ForbiddenAndConjunction(
                # Doesn't allow 3 consecutive maxpools with a large convolution mask in 3rd layer
                CS.ForbiddenEqualsClause(maxpool_1, 'True'),
                CS.ForbiddenEqualsClause(maxpool_2, 'True'),
                CS.ForbiddenInClause(kernel_3, ['3', '5', '7']),
                CS.ForbiddenEqualsClause(maxpool_3, 'True')
        )
        for_three_layers_8 = CS.ForbiddenAndConjunction(
                # Same as above, but stride=2 in place of maxpooling
                CS.ForbiddenEqualsClause(stride_1, 2),
                CS.ForbiddenEqualsClause(stride_2, 2),
                CS.ForbiddenInClause(kernel_3, ['3', '5', '7']),
                CS.ForbiddenEqualsClause(stride_3, 2)
        )
        for_three_layers_9 = CS.ForbiddenAndConjunction(
                # Allow a multiplication factor of only 0.5 for a 1x1 convolution in third layer
                # And no padding
                CS.ForbiddenInClause(kernel_3, ['3', '5', '7']),
                CS.ForbiddenInClause(channel_3, ['0.5']),
                CS.ForbiddenInClause(padding_3, [1, 2, 3])
        )
        for_three_layers_10 = CS.ForbiddenAndConjunction(
                # Allow a multiplication factor of only 0.5 for a 1x1 convolution in third layer
                # And no padding
                CS.ForbiddenEqualsClause(kernel_3, '1'),
                CS.ForbiddenInClause(channel_3, ['1', '2', '3']),
                CS.ForbiddenInClause(padding_3, [1, 2, 3])
        )
        for_three_layers_11 = CS.ForbiddenAndConjunction(
                # Disallowing large convolution filter following a large max pool
                CS.ForbiddenInClause(kernel_2, ['5', '7']),
                CS.ForbiddenEqualsClause(maxpool_2, 'True'),
                CS.ForbiddenInClause(kernel_3, ['5', '7'])
        )
        for_three_layers_12 = CS.ForbiddenAndConjunction(
                # Disallowing large convolution filter following a large max pool
                CS.ForbiddenInClause(kernel_2, ['5', '7']),
                CS.ForbiddenEqualsClause(maxpool_1, 'True'),
                CS.ForbiddenInClause(kernel_3, ['5', '7'])
        )
        config_space.add_forbidden_clauses([for_two_layers_1, for_two_layers_2, for_two_layers_2,
                                            for_three_layers_1_0, for_three_layers_1_1, for_three_layers_1_2,
                                            for_three_layers_2, for_three_layers_3, for_three_layers_4,
                                            for_three_layers_5, for_three_layers_6, for_three_layers_7,
                                            for_three_layers_8, for_three_layers_9, for_three_layers_10,
                                            for_three_layers_11, for_three_layers_12])
        # Forbidding a large convolution mask in the last layers
        last_layer_mask_1 = CS.ForbiddenAndConjunction(
                CS.ForbiddenEqualsClause(n_conv_layer, 3),
                CS.ForbiddenEqualsClause(kernel_3, '7')
        )
        last_layer_mask_2 = CS.ForbiddenAndConjunction(
                CS.ForbiddenEqualsClause(n_conv_layer, 2),
                CS.ForbiddenEqualsClause(kernel_2, '7')
        )
        config_space.add_forbidden_clauses([last_layer_mask_1, last_layer_mask_2])

        # INTERMEDIATE FULLY CONNECTED LAYER PARAMS AND CONDITIONS (NOT OUTPUT LAYER)
        # Choosing min size as height/width of image, max as height x width of image
        # Max of 784 >>> output of the convolutions and pooling, hence adequately expressed
        fc1 = CSH.UniformIntegerHyperparameter('fc_1', lower=28, upper=784, default_value=500, log=True)
        fc2 = CSH.UniformIntegerHyperparameter('fc_2', lower=28, upper=784, default_value=500, log=True)
        config_space.add_hyperparameters([fc1, fc2])
        # FC layers exists only if n_fc_layer > 1
        fc1_cond = CS.InCondition(fc1, n_fc_layer, [2,3])
        fc2_cond = CS.EqualsCondition(fc2, n_fc_layer, 3)
        config_space.add_conditions([fc1_cond, fc2_cond])

        return(config_space)


def testWorker(budget, test=False, save=None):
    '''
    A simple function to debug if the compute() from the worker interfaces the Target Algorithm correctly
    :param budget: Manually enter a budget (generally equivalent to min_budget in BOHB)
    :param test, save: Additional parameters to the Target Algorithm
    :return: The dict returned by compute()
    '''
    worker = MyWorker(run_id = 'cnn_bohb')
    config = worker.get_configspace().sample_configuration()
    print(config)
    res = worker.compute(config = config, budget = budget, test=test, save=save)
    return(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dataset", dest="dataset", type=str.upper, default='KMNIST', choices=['KMNIST, K49'],
                        help='Which dataset to evaluate on {K49, KMNIST}')
    # parser.add_argument("-w", "--n_workers", dest="n_workers", type=int, default=1,
    #                     help='Number of workers that will run')
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
    parser.add_argument('-r', "--run_id", dest="run_id", type=str, default='cnn_bohb',
                        help='Any specific run ID for annotation')
    parser.add_argument('-s', "--show_plots", dest="show_plots", type=bool, choices=[True, False], default=False,
                        help='To decide if plots additionally need to be opened in additional windows')
    args, kwargs = parser.parse_known_args()

    start_time = time.time()

    # Another historic relic - started experimentation with dataset having global scope
    # Should ideally be a part of *kwargs in compute
    # Luckily, the worker class is isolated and independent hence the global scope is not troublesome
    global dataset
    dataset = args.dataset # Find way to pass to BOHB call sans config

    # Starting server to communicate between target algorithm and BOHB
    NS = hpns.NameServer(run_id=args.run_id, host='127.0.0.1', port=None)
    NS.start()
    # Initialising the worker class (only one worker)
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
                  random_fraction=0.1,
                  num_samples=4)
    res = bohb.run(n_iterations=args.n_iterations )
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    print('='*40)
    print("Time taken for BOHB: ", time.time() - start_time)
    print('='*40)
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
    print(res.get_runs_by_id(incumbent)[-1]['info'])
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

    # End of main