import os
import argparse
import logging
import time

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from cnn import ConfigurableNet
from datasets import KMNIST, K49


def eval(model, loader, device, train=False):
    """
    Evaluation method
    :param model: Model to evaluate
    :param loader: data loader for either training or testing set
    :param device: torch device
    :param train: boolean to indicate if training or test set is used
    :return: accuracy on the data
    """
    true, pred = [], []
    with torch.no_grad():  # no gradient needed
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            true.extend(labels)
            pred.extend(predicted)
        # return balanced accuracy where each sample is weighted according to occurrence in dataset
        score = balanced_accuracy_score(true, pred)
        str_ = 'rain' if train else 'est'
        logging.info('T{0} Accuracy of the model on the {1} t{0} images: {2}%'.format(str_, len(true), 100 * score))
    return score


def train(dataset,
          model_config,
          data_dir,
          num_epochs=10,
          batch_size=50,
          learning_rate=0.001,
          train_criterion=torch.nn.CrossEntropyLoss,
          model_optimizer=torch.optim.Adam,
          data_augmentations=None,
          save_model_str=None):
    """
    Training loop for configurableNet.
    :param dataset: which dataset to load (str)
    :param model_config: configurableNet config (dict)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param train_criterion: Which loss to use during training (torch.nn._Loss)
    :param model_optimizer: Which model optimizer to use during trainnig (torch.optim.Optimizer)
    :param data_augmentations: List of data augmentations to apply such as rescaling.
        (list[transformations], transforms.Composition[list[transformations]], None)
        If none only ToTensor is used
    :return:
    """
    train_criterion = train_criterion()  # not instantiated until now

    # Device configuration (fixed to cpu as we don't provide GPUs for the project)
    device = torch.device('cpu')  # 'cuda:0' if torch.cuda.is_available() else 'cpu')

    # data_augmentations=transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.RandomRotation(15),
    #         # transforms.Resize((28,28)),
    #         # transforms.RandomAffine(degrees=15, translate=(0, 0.2), scale=(0.8,1.2), shear=10),
    #         transforms.ToTensor()
    # ])

    # data_augmentations = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
    #     transforms.RandomChoice([transforms.Resize((28, 28)),
    #                             transforms.RandomAffine(degrees=15, translate=(0,0.2),
    #                                                     scale=(0.8,1.2), shear=10)]),
    #     transforms.ToTensor()
    # ])


    if data_augmentations is None:
        # We only use ToTensor here as that is al that is needed to make it work
        data_augmentations = transforms.ToTensor()
    elif isinstance(type(data_augmentations), list):
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError

    if dataset == 'KMNIST':
        train_dataset = KMNIST(data_dir, True, data_augmentations)
        test_dataset = KMNIST(data_dir, False, data_augmentations)
    elif dataset == 'K49':
        train_dataset = K49(data_dir, True, data_augmentations)
        test_dataset = K49(data_dir, False, data_augmentations)
    else:
        raise NotImplementedError

    # Make data batch iterable
    # Could modify the sampler to not uniformly random sample
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    model = ConfigurableNet(model_config,
                            num_classes=train_dataset.n_classes,
                            height=train_dataset.img_rows,
                            width=train_dataset.img_cols,
                            channels=train_dataset.channels).to(device)
    total_model_params = np.sum(p.numel() for p in model.parameters())

    equal_freq = [1 / train_dataset.n_classes for _ in range(train_dataset.n_classes)]
    logging.debug('Train Dataset balanced: {}'.format(np.allclose(test_dataset.class_frequency, equal_freq)))
    logging.debug(' Test Dataset balanced: {}'.format(np.allclose(test_dataset.class_frequency, equal_freq)))
    logging.info('Generated Network:')
    summary(model, (train_dataset.channels, train_dataset.img_rows, train_dataset.img_cols), device='cpu')

    # Train the model
    optimizer = model_optimizer(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    train_time = time.time()
    epoch_times = []
    for epoch in range(num_epochs):
        logging.info('#' * 120)
        epoch_start_time = time.time()
        for i_batch, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward -> Backward <- passes
            outputs = model(images)
            loss = train_criterion(outputs, labels)
            optimizer.zero_grad()  # zero out gradients for new minibatch
            loss.backward()

            optimizer.step()
            if (i_batch + 1) % 100 == 0:
                logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, i_batch + 1, total_step, loss.item()))
        epoch_times.append(time.time() - epoch_start_time)
    train_time = time.time() - train_time

    # Test the model
    logging.info('~+~' * 40)
    model.eval()
    test_time = time.time()
    train_score = eval(model, train_loader, device, train=True)
    test_score = eval(model, test_loader, device)
    test_time = time.time() - test_time
    if save_model_str:
        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        if os.path.exists(save_model_str):
            save_model_str += '_'.join(time.ctime())
        torch.save(model.state_dict(), save_model_str)
    return train_score, test_score, train_time, test_time, total_model_params, model


if __name__ == '__main__':
    """
    This is just an example of how you can use train and evaluate to interact with the configurable network
    """
    loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss,
                 'mse': torch.nn.MSELoss}
    opti_dict = {'adam': torch.optim.Adam,
                 'adad': torch.optim.Adadelta,
                 'sgd': torch.optim.SGD}

    cmdline_parser = argparse.ArgumentParser('ML4AAD final project')

    cmdline_parser.add_argument('-d', '--dataset',
                                default='KMNIST',
                                help='Which dataset to evaluate on.',
                                choices=['KMNIST', 'K49'],
                                type=str.upper)
    cmdline_parser.add_argument('-e', '--epochs',
                                default=10,
                                help='Number of epochs',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=100,
                                help='Batch size',
                                type=int)
    cmdline_parser.add_argument('-D', '--data_dir',
                                default='../data',
                                help='Directory in which the data is stored (can be downloaded)')
    cmdline_parser.add_argument('-l', '--learning_rate',
                                default=0.001,
                                help='Optimizer learning rate',
                                type=float)
    cmdline_parser.add_argument('-L', '--training_loss',
                                default='cross_entropy',
                                help='Which loss to use during training',
                                choices=list(loss_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-o', '--optimizer',
                                default='adam',
                                help='Which optimizer to use during training',
                                choices=list(opti_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-m', '--model_path',
                                default=None,
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    train(
        args.dataset,  # dataset to use
        {  # model architecture
            # 'n_layers': 2,
            'n_fc_layer': 1,
            'n_conv_layer': 1,
            'dropout': 'False',
            'batchnorm': 'False',
            'channel_1': 3,
            'padding_1': 2,
            'stride_1': 1,
            'kernel_1': 5,
            'maxpool_1': 'True',
            'maxpool_kernel_1': 6,
            'activation': 'tanh'
        },
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_criterion=loss_dict[args.training_loss],
        model_optimizer=opti_dict[args.optimizer],
        data_augmentations=None,  # Not set in this example
        save_model_str=args.model_path
    )
