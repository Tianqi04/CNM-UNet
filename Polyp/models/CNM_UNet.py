import torch
import torch.nn as nn
import torch.nn.functional as F
import configs.config_setting as cfg
from models.Solver import nmNet_head_link_d1d2_2
from utils_DUSE.convert import AdaBN


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CNM_Block(nn.Module):
    def __init__(self,delta,in_channel1,in_channel2,in_channel3,in_channel4,in_channel5,out_channel,kernel_size=3,stride=1,padding=0,dilation=1):
        super().__init__()

        self.g1 = nn.Conv2d(in_channels=in_channel1, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                           padding=padding, dilation=dilation)
        self.g2 = nn.Conv2d(in_channels=in_channel2, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation)
        self.g3 = nn.Conv2d(in_channels=in_channel3, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation)
        self.g4 = nn.Conv2d(in_channels=in_channel4, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation)
        self.g5 = nn.Conv2d(in_channels=in_channel5, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation)

        self.sigma = nn.Sequential(
            nn.Softplus(),
            nn.BatchNorm2d(out_channel)
        )
        self.sigma2 = nn.Sequential(nn.BatchNorm2d(out_channel))
        # Continuous ODE solver
        self.Con_solver= nmNet_head_link_d1d2_2(delta, 3, 3,
                                                3)

    def forward(self, gamma, x1, x2, x3, x4, x5, b, weight_y1,
                weight_x1, weight_x2, weight_x3, weight_x4, weight_x5,
                scale1, scale2, scale3, scale4, scale5):
        y_next = self.sigma(weight_y1 * b +
                            F.interpolate(self.g1(weight_x1*x1),scale_factor=scale1,mode='bilinear') +
                            F.interpolate(self.g2(weight_x2*x2),scale_factor=scale2,mode='bilinear') +
                            F.interpolate(self.g3(weight_x3*x3),scale_factor=scale3,mode='bilinear') +
                            F.interpolate(self.g4(weight_x4*x4),scale_factor=scale4,mode='bilinear') +
                            F.interpolate(self.g5(weight_x5*x5),scale_factor=scale5,mode='bilinear'))


        B_y = y_next.shape[0] # y_next: [B,c,h,w]
        B_dydt = gamma.shape[0] # dydt: [1,c,h,w]
        if B_y != B_dydt:
            gamma_avg = gamma.expand(B_y, -1, -1, -1)
        else:
            gamma_avg = gamma

        if cfg.DUSE == False or self.training:
            y_next_t = self.sigma2(self.Con_solver(gamma_avg, y_next))
        else: # DUSE testing
            with torch.enable_grad():
                self.train()
                # Keep a copy of the original parameters.
                original_Solver_params = [param.data.clone() for param in self.Con_solver.parameters()]

                for param in self.Con_solver.parameters():
                    param.requires_grad_(True)
                optimizer_Con_Solver = torch.optim.Adam(self.Con_solver.parameters(),
                                                 lr=0.01,
                                                 betas=(0.9, 0.99),
                                                 weight_decay=0.00)
                y_next_t = self.sigma2(self.Con_solver(gamma_avg, y_next))
                y_next.requires_grad_(True)
                y_next_t.requires_grad_(True)

                for param in self.Con_solver.parameters():
                    param.requires_grad_(True)

                loss = self.Con_solver.dydt_loss
                loss.requires_grad_(True)

                optimizer_Con_Solver.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_Con_Solver.step()

                self.eval()

                y_next.grad = None
                y_next_t.grad = None
                for param in self.Con_solver.parameters():
                    param.requires_grad = False
            y_next_t = self.sigma2(self.Con_solver(gamma_avg, y_next))
            # Restore the original parameters.
            for param, saved_data in zip(self.Con_solver.parameters(), original_Solver_params):
                param.data.copy_(saved_data)

        return y_next_t

class CNM_UNet(nn.Module):
    def __init__(self, activefunc,droprate,kernel_size,n_channels, n_classes, bilinear=True):
        super(CNM_UNet, self).__init__()

        if activefunc == 'relu':
            self.act = nn.ReLU()
        elif activefunc == 'gelu':
            self.act = nn.GELU()
        elif activefunc == 'tanh':
            self.act = nn.Tanh()
        elif activefunc == 'softplus':
            self.act = nn.Softplus()
        self.drop = nn.Dropout(p=droprate)
        self.ker = kernel_size  #
        self.pad = (self.ker - 1) // 2

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # CNM-Block with Multiple Input Branches
        self.Mult_up = CNM_Block(delta=cfg.delta_t, in_channel1=64, in_channel2=128, in_channel3=256,
                                 in_channel4=512, in_channel5=512, out_channel=cfg.y0_channel, kernel_size=self.ker, stride=1, padding=self.pad)


        self.outc = OutConv(cfg.y0_channel, n_classes)

        self.Gamma = nn.Parameter(torch.zeros(1, cfg.y0_channel, cfg.input_size_h, cfg.input_size_w))
        self.newBN = AdaBN

    def change_BN_status(self, new_sample=True):
        flag = 0
        for nm, m in self.named_modules():
            if isinstance(m, self.newBN):
                m.new_sample = new_sample
                flag += 1
        return flag


    def forward(self, x):
        if (cfg.b_para):
            b = nn.Parameter(torch.zeros(x.shape[0], cfg.y0_channel, x.shape[2], x.shape[3])).cuda()
        else:
            b = torch.zeros(x.shape[0], cfg.y0_channel, x.shape[2], x.shape[3]).cuda()
        if (cfg.input_weight):
            input_weight = nn.Parameter(torch.ones(3, 2)).cuda()
        else:
            input_weight = torch.ones(3, 2).cuda()


        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        y_out = self.act(self.Mult_up(self.Gamma, x1, x2, x3, x4, x5, b, input_weight[0][0],
                                      input_weight[0][1], input_weight[1][0], input_weight[1][1], input_weight[2][0], input_weight[2][1],
                                      1, 2, 4, 8, 16))
        y_out = self.drop(y_out)

        y = self.outc(y_out)

        return torch.sigmoid(y)


if __name__ == '__main__':

    net = CNM_UNet(activefunc='softplus', droprate=0.1, kernel_size=3, n_channels=3, n_classes=1).cuda()
    from thop import profile


    dummy_input = torch.randn(1, 3, 352, 352).cuda()
    flops, params = profile(net, (dummy_input,))
    print('flops: %.2f M, params: %.2f k' % (flops / 1000000, params / 1000))
    print('net total parameters:', sum(param.numel() for param in net.parameters()))
