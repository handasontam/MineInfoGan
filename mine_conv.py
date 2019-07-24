import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def _resample(data, batch_size, replace=False):
    # Resample the given data sample.
    index = np.random.choice(
        range(data.shape[0]), size=batch_size, replace=replace)
    batch = data[index]
    return batch


def _div(net, data, ref):
    # Calculate the divergence estimate using a neural network
    mean_f = net(data).mean()
    log_mean_ef_ref = torch.logsumexp(net(ref), 0) - np.log(ref.shape[0])
    return mean_f - log_mean_ef_ref

class Net(nn.Module):
    def __init__(self, input_size=2, hidden_size=100, sigma=0.02):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, std=sigma)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=sigma)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=sigma)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

class MineConv(nn.Module):
    r"""Class for Mutual Information Neural Estimation. 

    The mutual information is estimated using neural estimation of the divergence 
    from joint distribution to product of marginal distributions.

    Arguments:
    ma_rate (float, optional): rate of moving average in the gradient estimate
    ma_ef (float, optional): initial value used in the moving average
    lr (float, optional): learning rate
    hidden_size (int, optional): size of the hidden layers
    """

    def __init__(self, channels, img_size, code_size, discrete_code_size, ma_rate=0.1, hidden_size=100, ma_ef=1):
        super(MineConv, self).__init__()
        self.ma_rate = ma_rate
        ds_size = img_size // 2 ** 4
        self.XY_net = Net(input_size=128 * ds_size ** 2 + code_size + discrete_code_size, 
                          hidden_size=300)

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        self.ma_ef = ma_ef  # for moving average

    def forward(self, img, code, discrete_code, img_marginal, code_marginal, discrete_code_marginal):
        r"""Train the neural networks for one or more steps.

        Argument:
        iter (int, optional): number of steps to train.
        """

        X = self.conv_blocks(img)
        X = X.view(X.shape[0], -1)
        Y_1 = code
        Y_2 = discrete_code
        batch_XY = torch.cat((X, Y_1, Y_2), dim=1)

        X_ref = self.conv_blocks(img_marginal)
        X_ref = X.view(X_ref.shape[0], -1)
        Y_ref_1 = code_marginal
        Y_ref_2 = discrete_code_marginal
        batch_XY_ref = torch.cat((X_ref, Y_ref_1, Y_ref_2), dim=1)
        
        # define the loss function with moving average in the gradient estimate
        mean_fXY = self.XY_net(batch_XY).mean()
        mean_efXY_ref = torch.exp(self.XY_net(batch_XY_ref)).mean()
        self.ma_ef = (1-self.ma_rate)*self.ma_ef + \
            self.ma_rate*mean_efXY_ref
        batch_loss_XY = - mean_fXY + \
            (1 / self.ma_ef.mean()).detach() * mean_efXY_ref
        return batch_loss_XY


    def state_dict(self):
        r"""Return a dictionary storing the state of the estimator.
        """
        return {
            'XY_net': self.XY_net.state_dict(),
            'ma_rate': self.ma_rate,
            'ma_ef': self.ma_ef
        }

    def load_state_dict(self, state_dict):
        r"""Load the dictionary of state state_dict.
        """
        self.XY_net.load_state_dict(state_dict['XY_net'])
        self.ma_rate = state_dict['ma_rate']
        self.ma_ef = state_dict['ma_ef']
