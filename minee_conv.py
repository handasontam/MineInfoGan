import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def _div(net, data, ref):
    # Calculate the divergence estimate using a neural network
    mean_f = net(data).mean()
    log_mean_ef_ref = torch.logsumexp(net(ref), 0) - np.log(ref.shape[0])
    return mean_f - log_mean_ef_ref

class Net(nn.Module):
    # Inner class that defines the neural network architecture
    def __init__(self, input_size=2, hidden_size=100, sigma=0.02):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, std=sigma)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=sigma)
        nn.init.constant_(self.fc2.bias, 0)
    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = self.fc2(output)
        return output

class MineeConv(nn.Module):
    r"""Class for Mutual Information Neural Entropic Estimation. 
    The mutual information is estimated using neural estimation of divergences 
    to uniform reference distribution.
    Arguments:
    X (tensor): samples of X
        dim 0: different samples
        dim 1: different components
    Y (tensor): samples of Y
        dim 0: different samples
        dim 1: different components
    ref_batch_factor (float, optional): multiplicative factor to increase 
        reference sample size relative to sample size
    lr (float, optional): learning rate
    hidden_size (int, optional): size of the hidden layers
    """


    def __init__(self, channels, img_size, code_size, discrete_code_size, hidden_size=100, cat_embed_dimension=2):
        super(MineeConv, self).__init__()
        ds_size = img_size // 2 ** 4
        self.XY_net = Net(input_size=128 * ds_size ** 2 + code_size + cat_embed_dimension, 
                          hidden_size=100)
        self.X_net = Net(input_size=128 * ds_size ** 2, 
                         hidden_size=100)
        # self.Y_net = Net(input_size=code_size + cat_embed_dimension, 
        #                  hidden_size=100)
        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.ReLU(inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        self.emb_layer = nn.Linear(discrete_code_size, cat_embed_dimension, bias=False)
        nn.init.normal_(self.emb_layer.weight, std=0.02)

    def forward(self, img, code, discrete_code, img_marginal, code_marginal, discrete_code_marginal):
        batch_X = self.conv_blocks(img)
        batch_X = batch_X.view(batch_X.shape[0], -1)
        Y_1 = code
        Y_2 = discrete_code
        Y_2 = self.emb_layer(Y_2)
        batch_XY = torch.cat((batch_X, Y_1, Y_2), dim=1)


        batch_X_ref = self.conv_blocks(img_marginal)
        batch_X_ref = batch_X_ref.view(batch_X_ref.shape[0], -1)
        Y_ref_1 = code_marginal
        Y_ref_2 = discrete_code_marginal
        Y_ref_2 = self.emb_layer(Y_ref_2)
        batch_XY_ref = torch.cat((batch_X_ref, Y_ref_1, Y_ref_2), dim=1)

        batch_loss_XY = -_div(self.XY_net, batch_XY, batch_XY_ref)
        batch_loss_X = -_div(self.X_net, batch_X, batch_X_ref)
        return batch_loss_X, batch_loss_XY

    def state_dict(self):
        r"""Return a dictionary storing the state of the estimator.
        """
        return {
            'XY_net': self.XY_net.state_dict(),
            'X_net': self.X_net.state_dict(),
            'Y_net': self.Y_net.state_dict(),
        }

    def load_state_dict(self, state_dict):
        r"""Load the dictionary of state state_dict.
        """
        self.XY_net.load_state_dict(state_dict['XY_net'])
        self.X_net.load_state_dict(state_dict['X_net'])
        self.Y_net.load_state_dict(state_dict['Y_net'])