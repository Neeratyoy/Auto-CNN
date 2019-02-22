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

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

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
        training_loss = torch.nn.CrossEntropyLoss # loss_dict[config['training_criterion']]
        # if config['training_criterion'] == 'MSELoss':
        #         training_loss = torch.nn.MSELossid2conf = result.get_id2config_mapping()
        # else:
        #     training_loss = torch.nn.CrossEntropyLoss
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

        train_score, train_loss, test_score, test_loss, train_time, test_time, total_model_params, _, _ = train(
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
            opti_aux_param=opti_aux_param,
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
        # loss = CSH.CategoricalHyperparameter('training_criterion', choices=['cross_entropy'], default_value='cross_entropy') # choices=['mse', 'cross_entropy']
        # batch = CSH.UniformIntegerHyperparameter('batch_size', lower=32, upper=1024, default_value=128)
        batch = CSH.CategoricalHyperparameter('batch_size', choices=['50', '100', '200', '500', '1000'], default_value='100')
        # ^ https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
        # ^ https://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent
        config_space.add_hyperparameters([batch])

        ############################
        # ARCHITECTURE HYPERPARAMS #
        ############################
        n_conv_layer = CSH.UniformIntegerHyperparameter('n_conv_layer', lower=1, upper=3, default_value=1, log=False)
        n_fc_layer = CSH.UniformIntegerHyperparameter('n_fc_layer', lower=1, upper=3, default_value=1, log=False)
        dropout = CSH.CategoricalHyperparameter('dropout', choices=['True', 'False'], default_value='False')
        activation = CSH.CategoricalHyperparameter('activation', choices=['relu', 'tanh', 'sigmoid'], default_value='tanh')
        batchnorm = CSH.CategoricalHyperparameter('batchnorm', choices=['True', 'False'], default_value='False')
        config_space.add_hyperparameters([n_conv_layer, n_fc_layer, dropout, activation, batchnorm])
        # LAYER 1 PARAMS
        kernel_1 = CSH.CategoricalHyperparameter('kernel_1', choices=['3', '5', '7'], default_value='5')
        channel_1 = CSH.UniformIntegerHyperparameter('channel_1', lower=3, upper = 12, default_value=3)
        padding_1 = CSH.UniformIntegerHyperparameter('padding_1', lower=0, upper=3, default_value=2)
        stride_1 = CSH.UniformIntegerHyperparameter('stride_1', lower=1, upper=2, default_value=1)
        # stride_1 = CSH.Constant('stride_1', 1)
        # batchnorm_1 = CSH.CategoricalHyperparameter('batchnorm_1', choices=['True', 'False'], default_value='False')
        # activation_1 = CSH.CategoricalHyperparameter('activation_1', choices=['tanh', 'relu', 'sigmoid'], default_value='tanh')
        maxpool_1 = CSH.CategoricalHyperparameter('maxpool_1', choices=['True', 'False'], default_value='True')
        maxpool_kernel_1 = CSH.UniformIntegerHyperparameter('maxpool_kernel_1', lower=2, upper=6, default_value=6)
        # dropout_1 = CSH.UniformFloatHyperparameter('dropout_1', lower=0, upper=0.3, default_value=0)
        config_space.add_hyperparameters([kernel_1, padding_1, stride_1, maxpool_1, maxpool_kernel_1, channel_1]) #, dropout_1, activation_1, batchnorm_1])
        # LAYER 1 CONDITIONALS
        maxpool_cond_1 = CS.NotEqualsCondition(maxpool_1, stride_1, 2)   # Convolution with stride 2 is equivalent to Maxpool
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

        # LAYER 2 PARAMS
        kernel_2 = CSH.CategoricalHyperparameter('kernel_2', choices=['3', '5', '7'], default_value='5')
        # Channels for Layer 2 onwards is a multiplicative factor of previous layer's channel size
        channel_2 = CSH.CategoricalHyperparameter('channel_2', choices=['1', '2', '3', '4'], default_value='2')
        padding_2 = CSH.UniformIntegerHyperparameter('padding_2', lower=0, upper=3, default_value=2)
        stride_2 = CSH.UniformIntegerHyperparameter('stride_2', lower=1, upper=2, default_value=1)
        # batchnorm_2 = CSH.CategoricalHyperparameter('batchnorm_2', choices=['True', 'False'], default_value='False')
        # activation_2 = CSH.CategoricalHyperparameter('activation_2', choices=['tanh', 'relu', 'sigmoid'], default_value='tanh')
        maxpool_2 = CSH.CategoricalHyperparameter('maxpool_2', choices=['True', 'False'], default_value='True')
        maxpool_kernel_2 = CSH.UniformIntegerHyperparameter('maxpool_kernel_2', lower=2, upper=6, default_value=6)
        # dropout_2 = CSH.UniformFloatHyperparameter('dropout_2', lower=0, upper=0.3, default_value=0)
        config_space.add_hyperparameters([kernel_2, padding_2, stride_2, maxpool_2, maxpool_kernel_2, channel_2]) #, dropout_2, activation_2, batchnorm_2])
        # LAYER 2 CONDITIONALS
        maxpool_cond_2 = CS.NotEqualsCondition(maxpool_2, stride_2, 2)   # Convolution with stride 2 is equivalent to Maxpool
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
        kernel_2_cond = CS.InCondition(kernel_2, n_conv_layer, [2, 3])
        channel_2_cond = CS.InCondition(channel_2, n_conv_layer, [2, 3])
        padding_2_cond = CS.InCondition(padding_2, n_conv_layer, [2, 3])
        stride_2_cond = CS.InCondition(stride_2, n_conv_layer, [2, 3])
        # batchnorm_2_cond = CS.EqualsCondition(batchnorm_2, n_conv_layer, 3)
        # activation_2_cond = CS.InCondition(activation_2, n_conv_layer, [2, 3])
        # dropout_2_cond = CS.InCondition(dropout_2, n_conv_layer, [2, 3])
        maxpool_2_cond = CS.AndConjunction(CS.InCondition(maxpool_2, n_conv_layer, [2, 3]), maxpool_cond_2)
        maxpool_kernel_2_cond = CS.AndConjunction(CS.InCondition(maxpool_kernel_2, n_conv_layer, [2, 3]), maxpool_kernel_cond_2)
        config_space.add_conditions([kernel_2_cond, channel_2_cond, padding_2_cond, stride_2_cond, maxpool_2_cond, maxpool_kernel_2_cond]) #, dropout_2_cond, activation_2_cond, batchnorm_2_cond])

        # LAYER 3 PARAMS
        kernel_3 = CSH.CategoricalHyperparameter('kernel_3', choices=['1', '3', '5', '7'], default_value='5')
        # Channels for Layer 2 onwards is a multiplicative factor of previous layer's channel size
        # Also being the max convolution layer allowed, this allows for 1x1 convolution
        # Therefore, a downsampling of channel depth (factor of 0.5)
        channel_3 = CSH.CategoricalHyperparameter('channel_3', choices=['0.5', '1', '2', '3'], default_value='2')
        padding_3 = CSH.UniformIntegerHyperparameter('padding_3', lower=0, upper=3, default_value=2)
        stride_3 = CSH.UniformIntegerHyperparameter('stride_3', lower=1, upper=2, default_value=1)
        # batchnorm_3 = CSH.CategoricalHyperparameter('batchnorm_3', choices=['True', 'False'], default_value='False')
        # activation_3 = CSH.CategoricalHyperparameter('activation_3', choices=['tanh', 'relu', 'sigmoid'], default_value='tanh')
        maxpool_3 = CSH.CategoricalHyperparameter('maxpool_3', choices=['True', 'False'], default_value='True')
        maxpool_kernel_3 = CSH.UniformIntegerHyperparameter('maxpool_kernel_3', lower=2, upper=6, default_value=6)
        # dropout_3 = CSH.UniformFloatHyperparameter('dropout_3', lower=0, upper=0.3, default_value=0)
        config_space.add_hyperparameters([kernel_3, padding_3, stride_3, maxpool_3, maxpool_kernel_3, channel_3]) #, dropout_3, activation_3, batchnorm_3])
        # LAYER 3 CONDITIONALS
        maxpool_cond_3 = CS.NotEqualsCondition(maxpool_3, stride_3, 2)   # Convolution with stride 2 is equivalent to Maxpool
        maxpool_kernel_cond_3 = CS.EqualsCondition(maxpool_kernel_3, maxpool_3, 'True')
        # LAYER 3 - RESTRICTING PADDING RANGE
        # Ensuring a padding domain of {0, 1, ..., floor(n/2)} for kernel_size n'activation': 'tanh', 'batch_size': '100', 'batchnorm': 'False', 'channel_1': 12, 'dropout': 'False', 'kernel_1': '7', 'learning_rate': 1.800472708316064e-05, 'model_optimizer': 'sgd', 'n_conv_layer': 3, 'n_fc_layer': 1, 'padding_1': 1, 'stride_1': 1, 'channel_2': '2', 'channel_3': '2', 'kernel_2': '3', 'kernel_3': '3', 'maxpool_1': 'True', 'momentum': 0.03883795397698147, 'padding_2': 0, 'padding_3': 0, 'stride_2': 2, 'stride_3': 1, 'maxpool_3': 'False', 'maxpool_kernel_1': 6}
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
        kernel_3_cond = CS.EqualsCondition(kernel_3, n_conv_layer, 3)
        channel_3_cond = CS.EqualsCondition(channel_3, n_conv_layer, 3)
        padding_3_cond = CS.EqualsCondition(padding_3, n_conv_layer, 3)
        stride_3_cond = CS.EqualsCondition(stride_3, n_conv_layer, 3)
        # batchnorm_3_cond = CS.EqualsCondition(batchnorm_3, n_conv_layer, 3)
        # activation_3_cond = CS.EqualsCondition(activation_3, n_conv_layer, 3)
        # dropout_3_cond = CS.EqualsCondition(dropout_3, n_conv_layer, 3)
        maxpool_3_cond = CS.AndConjunction(CS.InCondition(maxpool_3, n_conv_layer, [2, 3]), maxpool_cond_3)
        maxpool_kernel_3_cond = CS.AndConjunction(CS.InCondition(maxpool_kernel_3, n_conv_layer, [2, 3]), maxpool_kernel_cond_3)
        config_space.add_conditions([kernel_3_cond, channel_3_cond, padding_3_cond, stride_3_cond, maxpool_3_cond, maxpool_kernel_3_cond]) #, dropout_3_cond, activation_3_cond, batchnorm_3_cond])

        # COMPLICATED ASSUMPTIONS MADE EMPIRICALLY TO IMPOSE CONSTRAINTS ON MAXPOOL SIZE
        # SUCH THAT THE CONFIGURATIONS SAMPLED BY THE CONFIGURATOR DOESN'T YIELD AN
        # ARCHITECTURE WITH SHAPE/DIMENSION MISMATCH
        # FOLLOWING ASSUMPTIONS WERE MADE:
        #   1) AT MAX 3 CONVOLUTION LAYERS CAN BE FORMED
        #   2) CONVOLUTION KERNEL SIZE DOMAIN : {3, 5, 7}
        #   3) EACH CONVOLUTION LAYER MAY OR MAY NOT HAVE A MAXPOOL LAYER
        #   4) MAXPOOL KERNEL SIZE DOMAIN : {2, 3, 4, 5, 6}
        #   5) A CONVOLUTION WITH STRIDE 2 IS EQUIVALENT TO MAXPOOL KERNEL & STRIDE 2
        #       ^ https://arxiv.org/pdf/1412.6806.pdf
        for_two_layers_1 = CS.ForbiddenAndConjunction(
                # Disallowing large maxpool kernel in first layer for a 2-layer convoluiton
                CS.ForbiddenEqualsClause(n_conv_layer, 2),
                CS.ForbiddenInClause(maxpool_kernel_1, [3,4,5,6]),
                CS.ForbiddenInClause(maxpool_kernel_2, [4,5,6])
                # CS.ForbiddenEqualsClause(maxpool_2, 'True')
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
                # CS.ForbiddenEqualsClause(maxpool_2, 'True')
        )
        for_three_layers_1_1 = CS.ForbiddenAndConjunction(
                # Constraining maxpool kernel sizes for a 3 layer convolution
                # Small maxpool kernel if subsequent layer contains another maxpool
                CS.ForbiddenEqualsClause(n_conv_layer, 3),
                CS.ForbiddenInClause(maxpool_kernel_2, [4,5,6])
                # CS.ForbiddenEqualsClause(maxpool_2, 'True')
        )
        for_three_layers_1_2 = CS.ForbiddenAndConjunction(
                # Constraining maxpool kernel sizes for a 3 layer convolution
                # Small maxpool kernel if subsequent layer contains another maxpool
                CS.ForbiddenEqualsClause(n_conv_layer, 3),
                CS.ForbiddenInClause(maxpool_kernel_3, [3, 4,5,6])
                # CS.ForbiddenEqualsClause(maxpool_2, 'True')
        )
        for_three_layers_2 = CS.ForbiddenAndConjunction(
                # Constraining maxpool kernel sizes for a 3 layer convolution
                # Small maxpool kernel if subsequent layer contains another maxpoo)l
                CS.ForbiddenEqualsClause(n_conv_layer, 3),
                CS.ForbiddenInClause(maxpool_kernel_1, [3,4,5,6]),
                CS.ForbiddenInClause(maxpool_kernel_3, [5,6]),
                # CS.ForbiddenEqualsClause(maxpool_3, 'True')
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
                                            for_three_layers_2, for_three_layers_3,
                                            for_three_layers_4, for_three_layers_5, for_three_layers_6,
                                            for_three_layers_7, for_three_layers_8, for_three_layers_9,
                                            for_three_layers_10, for_three_layers_11, for_three_layers_12])
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
        # fc_dropout_1 = CSH.UniformFloatHyperparameter('fc_dropout_1', lower=0, upper=0.7, default_value=0)
        fc2 = CSH.UniformIntegerHyperparameter('fc_2', lower=28, upper=784, default_value=500, log=True)
        # fc_dropout_2 = CSH.UniformFloatHyperparameter('fc_dropout_2', lower=0, upper=0.7, default_value=0)
        config_space.add_hyperparameters([fc1, fc2])
        fc1_cond = CS.InCondition(fc1, n_fc_layer, [2,3])
        fc2_cond = CS.EqualsCondition(fc2, n_fc_layer, 3)
        config_space.add_conditions([fc1_cond, fc2_cond])

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
    parser.add_argument("--eta", dest="eta", type=int, default=3)
    parser.add_argument("--out_dir", dest="out_dir", type=str, default='bohb/')
    parser.add_argument("--run_id", dest="run_id", type=str, default='cnn_bohb')
    parser.add_argument("--show_plots", dest="show_plots", type=bool, default=False)
    args, kwargs = parser.parse_known_args()

    start_time = time.time()

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
    # print('~+~' * 40)
    # print('~+~' * 40)
    # print('===' * 40)
    # print("BUILDING AND EVALUATING INCUMBENT CONFIGURATION ON FULL TRAINING AND TEST SETS")
    # print('===' * 40)
    # result = hpres.logged_results_to_HBS_result(args.out_dir)
    # id2conf = result.get_id2config_mapping()
    # inc_id = result.get_incumbent_id()
    # inc_config = id2conf[inc_id]['config']
    # w = MyWorker('evaluating')
    # res = w.compute(config=inc_config, budget=args.max_budget, test=True, save=None) #args.out_dir)
    # print('~+~' * 40)
    # # print("Training Accuracy: ", res['info']['train_score'])
    # # print("Test Accuracy: ", res['info']['test_score'])
    # print("Training Loss: ", res['info']['train_loss'])
    # print("Test Loss: ", res['info']['test_score'])
