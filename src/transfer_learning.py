import numpy
import time
import argparse
import logging
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import torch
from BOHB_plotAnalysis import generateLossComparison, generateViz
import main
import main_transfer_learning
from generate_result import plot_confusion_matrix

def load_config(dir):
    '''
    Given a directory where BOHB results have been logged, loads the incumbent configuration
    The directory needs to have 'config.json' and 'results.json' as BOHB outputs
    :param dir: Directory where BOHB results exist
    :return: JSON containing incumbent configuration
    '''
    result = hpres.logged_results_to_HBS_result(dir)
    id2conf = result.get_id2config_mapping()
    inc_id = result.get_incumbent_id()
    inc_config = id2conf[inc_id]['config']
    return inc_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', "--dataset_source", dest="dataset_source", type=str, default='KMNIST',
                        choices=['KMNIST', 'K49'], help='Dataset to initially train on')
    parser.add_argument('-d', "--dataset_dest", dest="dataset_dest", type=str, default='K49',
                        choices=['KMNIST', 'K49'], help='Dataset to finally train on')
    parser.add_argument('-c', "--config_dir", dest="config_dir", type=str,
                        help='Directory that has config.json and results.json from a BOHB run')
    parser.add_argument('-e', "--epochs_source", dest="epochs_source", type=int, default=1, choices=list(range(1,21)),
                        help='Number of epochs for training first dataset')
    parser.add_argument('-E', "--epochs_dest", dest="epochs_dest", type=int, default=1, choices=list(range(1,21)),
                        help='Number of epochs for training second dataset')
    parser.add_argument('-v', '--verbose', default='INFO', choices=['INFO', 'DEBUG'], help='verbosity')
    args, kwargs = parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    start_time = time.time()

    print('~+~'*40)
    print("TRAINING PARENT MODEL ON ", args.dataset_source)
    print('~+~'*40)
    # Training model with the first dataset
    global dataset
    dataset = args.dataset_source # Find way to pass to BOHB call sans config
    # Reading configuration for the model from BOHB's incumbent
    config = load_config(args.config_dir)
    # Setting up other parameters to call Target Algorithm based on incumbent
    if config['model_optimizer'] == 'adam':
        if config['amsgrad'] == 'True':
            opti_aux_param = True
        else:
            opti_aux_param = False
    elif config['model_optimizer'] == 'sgd':
        opti_aux_param = config['momentum']
    else:
        opti_aux_param = None
    opti_dict = {'adam': torch.optim.Adam, 'adad': torch.optim.Adadelta,
                'sgd': torch.optim.SGD}
    # Calling train() from main.py which will train on Training Set and evaluate finally on Test set
    train_score_1, _, test_score_1, _, _, _, _, model, _ = main.train(
        dataset=dataset,  # dataset_source: The first dataset to initialize the model with
        model_config=config,
        data_dir='../data',
        num_epochs=args.epochs_source,
        batch_size=int(config['batch_size']),
        learning_rate=config['learning_rate'],
        train_criterion=torch.nn.CrossEntropyLoss,
        model_optimizer=opti_dict[config['model_optimizer']],
        opti_aux_param=opti_aux_param,
        data_augmentations=None,  # Not set in this example
        save_model_str=None,
        test=True
    )
    print('\n'*5)
    checkpoint1 = time.time() - start_time
    start_time = time.time()
    print('~+~'*40)
    print("TRANSFERRING MODEL TO ", args.dataset_dest)
    print('~+~'*40)
    # Transfering model to the second dataset
    #   Only changes : 1) Model as an extra argument
    #                  2) No configuration passed
    #                  3) Different dataset being passed
    dataset = args.dataset_dest
    train_score_2, _, test_score_2, _, _, _, _, _, cm = main_transfer_learning.train(
        dataset=dataset, # Calling train() from main_transfer_learning.py which alters the output layer size
        old_model=model, # Learnt model from previous dataset
        # model_config=config, # No need of config since model (architecture) remains intact
        data_dir='../data',
        num_epochs=args.epochs_dest,
        batch_size=int(config['batch_size']),
        learning_rate=config['learning_rate'],
        train_criterion=torch.nn.CrossEntropyLoss,
        model_optimizer=opti_dict[config['model_optimizer']],
        opti_aux_param=opti_aux_param,
        data_augmentations=None, #True,  # Not set in this example
        save_model_str=None,
        test=True
    )
    # Plotting confusion matrix
    if args.dataset_dest == 'K49':
        plot_confusion_matrix(cm, range(0,49), out_dir='./', dataset=args.dataset_dest, name=args.epochs_dest, normalize=False)
    else:
        plot_confusion_matrix(cm, range(0,10), out_dir='./', dataset=args.dataset_dest, name=args.epochs_dest, normalize=False)
    print('~+~'*40)
    print("Training and test score for first phase of training on "+args.dataset_source)
    print(train_score_1, test_score_1)
    print("Time taken: "+str(checkpoint1))
    print("Training and test score for second phase of training on "+args.dataset_dest)
    print(train_score_2, test_score_2)
    print("Time taken: "+str(time.time() - start_time))
    print('~+~'*40)
    # end of main
