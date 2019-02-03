import numpy as np
import torch.nn as nn


# define ConvNet #######################################################################################################
class ConfigurableNet(nn.Module):
    """
    Example of a configurable network. (No dropout or skip connections supported)
    """
    def _update_size(self, dim, padding, dilation, kernel_size, stride):
        """
        Helper method to keep track of changing output dimensions between convolutions and Pooling layers
        returns the updated dimension "dim" (e.g. height or width)
        """
        return int(np.floor((dim + 2 * padding - dilation * (kernel_size - 1) + 1) / stride))

    def __init__(self, config, num_classes=10, height=28, width=28, channels=1):
        """
        [PyTorch syntax] The constructor: Declare all the layers to be used
        Configurable network for image classification
        :param config: network config to construct architecture with
        :param num_classes: Number of outputs required
        :param height: image height
        :param width: image width
        """
        super(ConfigurableNet, self).__init__()
        self.config = config

        # Keeping track of internals like changeing dimensions
        n_convs = config['n_conv_layer']
        conv_layer = 0
        self.layers = []
        self.mymodules = nn.ModuleList()
        out_channels = channels

        # Create sequential network
        for layer in range(config['n_layers']):
            if n_convs >= 1:  # This way it only supports multiple convolutional layers at the beginning (not inbetween)
                l = []  # Conv layer can be sequential layer with Batch Norm and pooling
                padding = 2
                stride = 1
                kernel_size = 5
                dilation = 1  # fixed
                if conv_layer == 0:
                    out_channels = 3
                else:
                    # instead of handling different widths for each conv layer, just per convolution add the same size
                    out_channels += 3

                # get convolution
                c = nn.Conv2d(channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)

                # update dimensions
                channels = out_channels
                height = self._update_size(height, padding, dilation, kernel_size, stride)
                width = self._update_size(width, padding, dilation, kernel_size, stride)
                l.append(c)

                # batchnorm yes or no?
                batchnorm = False
                if batchnorm:
                    b = nn.BatchNorm2d(channels)
                    l.append(b)

                # determine activation function,
                activation = 'tanh'
                if activation == 'relu':
                    act = nn.ReLU()
                elif activation == 'sigmoid':
                    act = nn.Sigmoid()
                elif activation == 'tanh':
                    act = nn.Tanh()
                else:
                    # Add more activation funcs?
                    raise NotImplementedError
                l.append(act)

                # do max pooling yes or no?
                max_pooling = True
                if max_pooling:
                    m_ks = 6
                    m_stride = 6
                    pool = nn.MaxPool2d(kernel_size=m_ks,
                                        stride=m_stride)
                    l.append(pool)
                    height = self._update_size(height, 0, 1, m_ks, m_stride)
                    width = self._update_size(width, 0, 1, m_ks, m_stride)
                n_convs -= 1
                conv_layer += 1

                # setup everything as sequential layer
                s = nn.Sequential(*l)
                self.mymodules.append(s)
                self.layers.append(s)

            # handle intermediate fully connected layers
            elif layer < config['n_layers'] - 1:
                if n_convs == 0:  # compute fully connected input size
                    channels = height * width * channels
                    n_convs -= 1
                    #           in_channels, out_channels
                lay = nn.Linear(channels, 500)
                self.mymodules.append(lay)
                self.layers.append(lay)
                channels = 500  # update the channels to keep track how many inputs lead to the next layer

            # handle final fully connected layer
            else:
                if n_convs == 0:
                    channels = height * width * channels
                    n_convs -= 1
                out = nn.Linear(channels, num_classes)
                self.mymodules.append(out)
                self.layers.append(out)

    def forward(self, out):
        '''
        [PyTorch syntax] Forward function: Defines how the model is going to be run, from input to output
        :param out:
        :return:
        '''
        for idx, layer in enumerate(self.layers):
            if self.config['n_conv_layer'] == idx:
                out = out.reshape(out.size(0), -1)  # flatten the output after convolutions (keeping batch dimension)
            out = layer(out)
        return out
