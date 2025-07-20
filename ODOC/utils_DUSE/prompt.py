import torch
import torch.nn as nn
import torch.nn.functional as F


class Prompt(nn.Module):
    def __init__(self, prompt_alpha=0.01, image_size=512,batch_size = 1):
        super().__init__()
        self.prompt_size = int(image_size * prompt_alpha) if int(image_size * prompt_alpha) > 1 else 1
        # 提示图在原始图像中的填充大小，将提示图居中放置在原始图像中
        # padding 1 around Pi to expand it to the size of H×W
        self.padding_size = (image_size - self.prompt_size)//2
        # 此处有修改
        self.init_para = torch.ones((batch_size, 3, self.prompt_size, self.prompt_size)) # (B,c,H,W)形式
        # 将init_para转换为可训练的参数，并赋值给self.data_prompt
        self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)
        self.pre_prompt = self.data_prompt.detach().cpu().data # 创建一个不可训练的提示图参数副本，用于后续的操作

    def update(self, init_data):
        with torch.no_grad():
            # init_data 的数据复制到 self.data_prompt中
            self.data_prompt.copy_(init_data)

    def iFFT(self, amp_src_, pha_src, imgH, imgW):
        # amp_src_（源图像的幅度谱）、pha_src（源图像的相位谱）、imgH（图像的高度）、imgW（图像的宽度）
        # recompose fft
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)

        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg

    def forward(self, x):
        _, _, imgH, imgW = x.size()

        # x.clone() 创建了输入的一个副本，以避免对原始数据的修改。dim=(-2, -1) 指定在最后两个维度（即高和宽）上计算 FFT
        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src) # 使用 fftshift 将幅度的零频率分量移至频谱中心,在处理中更好地观察频谱图

        # obtain the low frequency amplitude part
        # 使用 F.pad 对 self.data_prompt 进行填充，以确保它与输入图像的尺寸匹配
        prompt = F.pad(self.data_prompt, [self.padding_size, imgH - self.padding_size - self.prompt_size,
                                          self.padding_size, imgW - self.padding_size - self.prompt_size],
                       mode='constant', value=1.0).contiguous()

        amp_src_ = amp_src * prompt # 将提示图叠加到源图像的振幅谱中
        amp_src_ = torch.fft.ifftshift(amp_src_) # 使用 ifftshift 将幅度数据恢复为标准的频谱格式，以便于进行逆傅里叶变换

        # 从输入图像的幅度谱中提取出低频部分
        amp_low_ = amp_src[:, :, self.padding_size:self.padding_size+self.prompt_size, self.padding_size:self.padding_size+self.prompt_size]

        # 调用 iFFT 方法计算逆傅里叶变换，以生成图像空间中的更新图像 src_in_trg
        src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW)
        return src_in_trg, amp_low_
