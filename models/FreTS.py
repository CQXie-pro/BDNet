import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2311.06184.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.embed_size = 128  # embed_size
        self.hidden_size = 256  # hidden_size
        self.pred_len = configs.pred_len
        self.feature_size = configs.enc_in  # channels
        self.seq_len = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)  # B N T
        x = x.unsqueeze(3)  # B N T 1
        # N*T*1 x 1*D = N*T*D
        # self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        y = self.embeddings
        return x * y  # B N T D

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # x: [B, N, T, D]
        x = x.permute(0, 2, 1, 3)  # [B, T, N, D]

        # FFT on N dimension
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # [B, T, N/2+1, D]

        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)  # [B, T, N/2+1, D]

        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")  # [B, T, N, D]

        x = x.permute(0, 2, 1, 3)  # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        # 初始化输出张量
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size], device=x.device)  # [B, T, N/2+1, D]
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size], device=x.device)  # [B, T, N/2+1, D]

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )  # [B, T, N/2+1, D]

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )  # [B, T, N/2+1, D]

        y = torch.stack([o1_real, o1_imag], dim=-1)  # [B, T, N/2+1, D, 2]

        # 对组合后的张量应用Softshrink激活函数，使得小于某个阈值（self.sparsity_threshold）的值被压缩为零，以实现稀疏性。
        y = F.softshrink(y, lambd=self.sparsity_threshold)  # [B, T, N/2+1, D, 2]

        # 将实部和虚部的张量视为复数张量
        y = torch.view_as_complex(y)  # [B, T, N/2+1, D]
        return y

    def forecast(self, x_enc):
        # x: [Batch, Input length, Channel]
        B, T, N = x_enc.shape

        x = self.tokenEmb(x_enc)  # [B, N, T, D]
        bias = x

        if self.channel_independence == '0':
            x = self.MLP_channel(x, B, N, T)  # [B, N, T, D]

        x = self.MLP_temporal(x, B, N, T)  # [B, N, T, D]

        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)  # B, pre_len, N
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')
