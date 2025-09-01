import torch
import torch.nn as nn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class combined_series_decomp(nn.Module):
    def __init__(self, kernel_size, top_k=5):
        super(combined_series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.top_k = top_k

    def forward(self, x):
        # Compute the trend using moving average
        moving_mean = self.moving_avg(x)
        x = x - moving_mean

        # Compute the seasonality using DFT
        xf = torch.fft.rfft(x, dim=1)
        freq = abs(xf)
        freq[:, 0, :] = 0  # Ignore the DC component
        top_k_freq, _ = torch.topk(freq, self.top_k, dim=1)
        threshold = top_k_freq[:, -1, :].unsqueeze(1).expand_as(freq)
        xf[freq < threshold] = 0
        x_season = torch.fft.irfft(xf, dim=1)

        # Compute the residual (original series minus trend and seasonality)
        x_residual = x - x_season

        return x_season, moving_mean, x_residual

class Backbone(nn.Module):
    def __init__(self, configs):
        super(Backbone, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type

        if self.model_type == "linear":
            self.linear = nn.Linear(self.seq_len, self.pred_len)
            self.linear.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.input = nn.Linear(self.seq_len, self.d_model)
            self.activation = nn.ReLU()
            self.output = nn.Linear(self.d_model, self.pred_len)

    def forward(self, x):
        if self.model_type == "linear":
            x = self.linear(x)
        else:
            x = self.input(x)
            x = self.output(self.activation(x))
        return x

class Model(nn.Module):
    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = individual
        self.channels = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = nn.Dropout(configs.dropout)

        self.decompsition_block_1 = combined_series_decomp(24 + 1, top_k=5)
        self.decompsition_block_2 = combined_series_decomp(24 * 7 + 1, top_k=5)

        if self.individual:
            self.Linear_Seasonal_1 = nn.ModuleList()
            self.Linear_Trend_1 = nn.ModuleList()
            self.Linear_Res_1 = nn.ModuleList()

            self.Linear_Seasonal_2 = nn.ModuleList()
            self.Linear_Trend_2 = nn.ModuleList()
            self.Linear_Res_2 = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal_1.append(Backbone(configs))
                self.Linear_Trend_1.append(Backbone(configs))
                self.Linear_Res_1.append(Backbone(configs))

                self.Linear_Seasonal_2.append(Backbone(configs))
                self.Linear_Trend_2.append(Backbone(configs))
                self.Linear_Res_2.append(Backbone(configs))

        else:
            self.trend = Backbone(configs)
            self.season_1 = Backbone(configs)
            self.season_2 = Backbone(configs)
            self.res = Backbone(configs)

    def encoder(self, x, normal):
        # x: [B, L, D]
        # Normalization
        if normal:
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x = x - x_mean
            x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = x / torch.sqrt(x_var)

        # 针对多周期交通数据的多尺度特征提取技术，第一次分解提取日周期，第二次分解提取周周期
        season_1, trend_1, res_1 = self.decompsition_block_1(x)  # B, L, D
        season_2, trend_2, res_2 = self.decompsition_block_2(trend_1)  # B, L, D

        res = res_1 + res_2
        trend = trend_2

        season_1 = season_1.permute(0, 2, 1)
        season_2, trend, res = season_2.permute(0, 2, 1), trend.permute(0, 2, 1), res.permute(0, 2, 1)
        # Linear Layer
        if self.individual:
            season_1_output = torch.zeros([season_1.size(0), season_1.size(1), self.pred_len],
                                        dtype=season_1.dtype).to(season_1.device)
            trend_1_output = torch.zeros([trend_1.size(0), trend_1.size(1), self.pred_len],
                                       dtype=trend_1.dtype).to(trend_1.device)
            res_1_output = torch.zeros([res_1.size(0), res_1.size(1), self.pred_len],
                                     dtype=res_1.dtype).to(res_1.device)

            season_2_output = torch.zeros([season_2.size(0), season_2.size(1), self.pred_len],
                                          dtype=season_2.dtype).to(season_2.device)
            trend_2_output = torch.zeros([trend_2.size(0), trend_2.size(1), self.pred_len],
                                         dtype=trend_2.dtype).to(trend_2.device)
            res_2_output = torch.zeros([res_2.size(0), res_2.size(1), self.pred_len],
                                       dtype=res_2.dtype).to(res_2.device)
            for i in range(self.channels):
                season_1_output[:, i, :] = self.Linear_Seasonal_1[i](season_1[:, i, :])
                trend_1_output[:, i, :] = self.Linear_Trend_1[i](trend_1[:, i, :])
                res_1_output[:, i, :] = self.Linear_Res_1[i](res_1[:, i, :])

                season_2_output[:, i, :] = self.Linear_Seasonal_2[i](season_2[:, i, :])
                trend_2_output[:, i, :] = self.Linear_Trend_2[i](trend_2[:, i, :])
                res_2_output[:, i, :] = self.Linear_Res_2[i](res_2[:, i, :])

        else:
            season_1_output = self.season_1(season_1)  # B, D, pre_len
            season_2_output = self.season_2(season_2)  # B, D, pre_len
            trend_output = self.trend(trend)  # B, D, pre_len
            res_output = self.res(res)  # B, D, pre_len

        x = season_1_output + season_2_output + trend_output + res_output
        x = x.permute(0, 2, 1)

        # De-Normalization
        if normal:
            x = x * torch.sqrt(x_var) + x_mean

        return x  # [Batch, Output length, Channel]

    def forecast(self, x_enc, normal):
        # Encoder
        return self.encoder(x_enc, normal)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, normal=True):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, normal)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None