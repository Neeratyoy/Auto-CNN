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
        # return int(np.floor((dim + 2 * padding - dilation * (kernel_size - 1) + 1) / stride))
        return int(np.floor((dim + 2*padding - (dilation*(kernel_size-1) + 1))/stride + 1))
        # output_size=(w+2*pad-(d(k-1)+1))/s+1, https://github.com/vlfeat/matconvnet/issues/1010
        # ^ works for both maxpool=T and maxpool=F

    def __init__(self, old_config, new_config, num_classes=10, height=28, width=28, channels=1):
        """
        [PyTorch syntax] The constructor: Declare all the layers to be used
        Configurable network for image classification
        :param config: network config to construct architecture with
        :param num_classes: Number of outputs required
        :param height: image height
        :param width: image width
        """
        super(ConfigurableNet, self).__init__()
        self.config = old_config
        dropout_dict = {'True': True, 'False': False}
        dropout = dropout_dict[old_config['dropout']]
        batchnorm_dict = {'True':True, 'False':False}
        batchnorm = batchnorm_dict[old_config['batchnorm']]

        channel_dict = {'1': int(new_config['channel_1'])}
        for i in range(2, old_config['n_conv_layer']+1):
            # Multiplying the multiplicative factor with previous channel size
            channel_dict[str(i)] = int(np.floor(float(new_config['channel_'+str(i)]) * channel_dict[str(i-1)]))

        # Keeping track of internals like changeing dimensions
        n_convs = old_config['n_conv_layer']
        n_fc_layers = new_config['n_fc_layer']
        n_layers = n_convs + n_fc_layers
        conv_layer = 0
        self.layers = []
        self.mymodules = nn.ModuleList()
        # out_channels = channels
        out_channels = channel_dict['1']

        # Create sequential network
        for layer in range(n_layers):
            if n_convs >= 1:  # This way it only supports multiple convolutional layers at the beginning (not inbetween)
                l = []  # Conv layer can be sequential layer with Batch Norm and pooling
                padding = old_config['padding_'+str(layer+1)] # 5 0 #2
                stride = old_config['stride_'+str(layer+1)] # 5 0 #21
                kernel_size = int(old_config['kernel_'+str(layer+1)]) # 5
                dilation = 1  # fixed
                # if conv_layer == 0:
                #     out_channels = 3
                # else:
                #     # instead of handling different widths for each conv layer, just per convolution add the same size
                #     out_channels += 3
                out_channels = channel_dict[str(layer+1)]

                # get convolution
                c = nn.Conv2d(channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)

                # update dimensions
                channels = out_channels
                height = self._update_size(height, padding, dilation, kernel_size, stride)
                width = self._update_size(width, padding, dilation, kernel_size, stride)
                l.append(c)

                # batchnorm yes or no?
                # try:
                #     batchnorm = batchnorm_dict[config['batchnorm_'+str(layer+1)]]
                # except KeyError:
                #     batchnorm = config['batchnorm_'+str(layer+1)] = False
                # batchnorm = False
                if batchnorm:
                    b = nn.BatchNorm2d(channels)
                    l.append(b)

                # determine activation function,
                # activation = config['activation_'+str(layer+1)]
                activation = old_config['activation']
                # activation = 'tanh'
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

                # Adding Dropout
                if dropout:
                    l.append(nn.Dropout(0.2))
                    # l.append(nn.Dropout(config['dropout_'+str(layer+1)]))

                # do max pooling yes or no?
                maxpool_dict = {'True':True, 'False':False}
                try:
                    max_pooling = maxpool_dict[old_config['maxpool_'+str(layer+1)]]
                except KeyError:
                    max_pooling = False
                # max_pooling = True
                if max_pooling:
                    m_ks = old_config['maxpool_kernel_'+str(layer+1)] # 6
                    m_stride = m_ks #6
                    pool = nn.MaxPool2d(kernel_size=m_ks, stride=m_stride)
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
            elif layer < n_layers - 1:
                if n_convs == 0:  # compute fully connected input size
                    channels = height * width * channels
                    n_convs -= 1
                    #           in_channels, out_channels
                output_count = 500 #new_config['fc_'+str(layer+1-config['n_conv_layer'])]
                lay = []
                lay.append(nn.Linear(channels, output_count))
                if batchnorm:
                    b = nn.BatchNorm1d(output_count)
                    lay.append(b)
                # determine activation function
                activation = config['activation']
                # activation = 'tanh'
                if activation == 'relu':
                    act = nn.ReLU()
                elif activation == 'sigmoid':
                    act = nn.Sigmoid()
                elif activation == 'tanh':
                    act = nn.Tanh()
                else:
                    # Add more activation funcs?
                    raise NotImplementedError
                lay.append(act)
                if dropout:
                    lay.append(nn.Dropout(0.5))
                s = nn.Sequential(*lay)
                self.mymodules.append(s)
                self.layers.append(s)
                channels = output_count  # update the channels to keep track how many inputs lead to the next layer

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
