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
    result = hpres.logged_results_to_HBS_result(dir)
    id2conf = result.get_id2config_mapping()
    inc_id = result.get_incumbent_id()
    inc_config = id2conf[inc_id]['config']
    return inc_config

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_source", dest="dataset_source", type=str, default='KMNIST')
    parser.add_argument("--dataset_dest", dest="dataset_dest", type=str, default='K49')
    parser.add_argument("--config_dir", dest="config_dir", type=str, default='bohb/KMNIST/4_2_20/')
    parser.add_argument("--epochs_source", dest="epochs_source", type=int, default=1)
    parser.add_argument("--epochs_dest", dest="epochs_dest", type=int, default=1)
    args, kwargs = parser.parse_known_args()

    start_time = time.time()

    print('~+~'*40)
    print("TRAINING PARENT MODEL ON ", args.dataset_source)
    print('~+~'*40)

    # Training model with the first dataset
    global dataset
    dataset = args.dataset_source # Find way to pass to BOHB call sans config
    config = load_config(args.config_dir)
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
    train_score_1, _, test_score_1, _, _, _, _, model, _ = main.train(
        dataset=dataset,  # dataset to use
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

    print("Checkpoint 1: ", time.time() - start_time)
    start_time = time.time()

    print('~+~'*40)
    print("TRANSFERING MODEL TO ", args.dataset_dest)
    print('~+~'*40)
    # Transfering model to the second dataset
    #   Only changes : 1) Model as an extra argument
    #                  2) Diff()erent dataset being passed
    dataset = args.dataset_dest
    train_score_2, _, test_score_2, _, _, _, _, _, cm = main_transfer_learning.train(
        dataset=dataset, # dataset to use
        old_model=model, # learnt model from previous dataset
        # model_config=config,
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
    if args.dataset_dest == 'K49':
        plot_confusion_matrix(cm, range(0,49), out_dir='./', dataset=args.dataset_dest, name=args.epochs_dest, normalize=False)
    else:
        plot_confusion_matrix(cm, range(0,10), out_dir='./', dataset=args.dataset_dest, name=args.epochs_dest, normalize=False)

    print(train_score_1, test_score_1)
    print(train_score_2, test_score_2)
    print('~+~'*40)
    print("Checkpoint 2: ", time.time() - start_time)
    start_time = time.time()
