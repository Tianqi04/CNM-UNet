import torch.nn as nn
import numpy as np
from torch import nn as nn


class AdaBN(nn.BatchNorm2d):
    def __init__(self, in_ch, warm_n=5):
        super(AdaBN, self).__init__(in_ch)
        self.warm_n = warm_n
        self.sample_num = 0
        self.new_sample = False

    def get_mu_var(self, x):
        if self.new_sample:
            self.sample_num += 1
        C = x.shape[1]

        cur_mu = x.mean((0, 2, 3), keepdims=True).detach()
        cur_var = x.var((0, 2, 3), keepdims=True).detach()


        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)


        moment = 1 / ((np.sqrt(self.sample_num) / self.warm_n) + 1)

        new_mu = moment * cur_mu + (1 - moment) * src_mu
        new_var = moment * cur_var + (1 - moment) * src_var
        return new_mu, new_var

    def forward(self, x):
        N, C, H, W = x.shape

        new_mu, new_var = self.get_mu_var(x)

        cur_mu = x.mean((2, 3), keepdims=True)
        cur_std = x.std((2, 3), keepdims=True)
        self.bn_loss = (
                (new_mu - cur_mu).abs().mean() + (new_var.sqrt() - cur_std).abs().mean()
        )

        # Normalization with new statistics
        new_sig = (new_var + self.eps).sqrt()

        new_x = ((x - new_mu) / new_sig) * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        return new_x


def replace_bn_with_adabn(module, warm_n=5):
    """
    Recursively replace BatchNorm layers with custom AdaBN.
    """
    flag = 0
    for name, child in list(module.named_children()):
        #if isinstance(child, nn.BatchNorm2d):
        if isinstance(child, nn.BatchNorm2d) and not isinstance(child, AdaBN):
            # create AdaBN instance and copy parameters from BatchNorm
            new_norm = AdaBN(child.num_features, warm_n=warm_n).to(child.weight.device)

            if hasattr(new_norm, 'load_old_dict'):

                new_norm.load_old_dict(child)
                msg = f"Converted {name} to {type(new_norm).__name__} (custom loader)"
            elif hasattr(new_norm, 'load_state_dict'):

                state_dict = child.state_dict()
                load_result = new_norm.load_state_dict(state_dict, strict=True)
                msg = f"Converted {name} to {type(new_norm).__name__} (standard loader) Missing keys: {load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}"
            else:
                msg = 'No load_old_dict() found!!!'


            setattr(module, name, new_norm)
            flag += 1
            print(msg)
        else:
            # Replace the existing BatchNorm modules.
            flag += replace_bn_with_adabn(child, warm_n=warm_n)

    return flag
