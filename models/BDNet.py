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


class BD(nn.Module):
    def __init__(self, kernel_size, top_k=5):
        super(BD, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.top_k = top_k

    def forward(self, x):
        # Compute the trend using moving average
        x_trend = self.moving_avg(x)
        x = x - x_trend

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

        return x_season, x_trend, x_residual


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


class BDNet_BD(nn.Module):
    def __init__(self, configs):
        super(BDNet_BD, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.d_model = configs.d_model
        self.moving_avg = configs.moving_avg
        self.individual = configs.channel_independence

        self.BD = BD(self.moving_avg, top_k=configs.top_k)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Res = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(Backbone(configs))
                self.Linear_Trend.append(Backbone(configs))
                self.Linear_Res.append(Backbone(configs))
        else:
            self.trend = Backbone(configs)
            self.season = Backbone(configs)
            self.res = Backbone(configs)

    def forward(self, x):
        # x: [B, L, D]
        season, trend, res = self.BD(x)  # B, L, D
        season, trend, res = season.permute(0, 2, 1), trend.permute(0, 2, 1), res.permute(0, 2, 1)

        # Linear Layer
        if self.individual:
            season_output = torch.zeros([season.size(0), season.size(1), self.pred_len], dtype=season.dtype).to(season.device)
            trend_output = torch.zeros([trend.size(0), trend.size(1), self.pred_len], dtype=trend.dtype).to(trend.device)
            res_output = torch.zeros([res.size(0), res.size(1), self.pred_len], dtype=res.dtype).to(res.device)
            for i in range(self.channels):
                season_output[:, i, :] = self.Linear_Seasonal[i](season[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend[:, i, :])
                res_output[:, i, :] = self.Linear_Trend[i](res[:, i, :])
        else:
            season_output = self.season(season)  # B, D, P
            trend_output = self.trend(trend)
            res_output = self.res(res)

        x = season_output + trend_output + res_output
        x = x.permute(0, 2, 1)

        return x  # B, P, D


class BDNet_BDS(nn.Module):
    def __init__(self, configs):
        super(BDNet_BDS, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.d_model = configs.d_model
        self.moving_avg = configs.moving_avg
        self.individual = configs.channel_independence

        self.BD_d = BD(24 + 1, top_k=5)
        self.BD_w = BD(24 * 7 + 1, top_k=5)

        if self.individual:
            self.Linear_Seasonal_d = nn.ModuleList()
            self.Linear_Trend_d = nn.ModuleList()
            self.Linear_Res_d = nn.ModuleList()

            self.Linear_Seasonal_w = nn.ModuleList()
            self.Linear_Trend_w = nn.ModuleList()
            self.Linear_Res_w = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal_d.append(Backbone(configs))
                self.Linear_Trend_d.append(Backbone(configs))
                self.Linear_Res_d.append(Backbone(configs))

                self.Linear_Seasonal_w.append(Backbone(configs))
                self.Linear_Trend_w.append(Backbone(configs))
                self.Linear_Res_w.append(Backbone(configs))

        else:
            self.trend_d = Backbone(configs)
            self.season_d = Backbone(configs)
            self.res_d = Backbone(configs)

            self.trend_w = Backbone(configs)
            self.season_w = Backbone(configs)
            self.res_w = Backbone(configs)

    def forward(self, x):
        # x: [B, L, D]
        season_d, trend_d, res_d = self.BD_d(x)  # B, L, D
        season_w, trend_w, res_w = self.BD_w(trend_d)  # B, L, D

        season_d, trend_d, res_d = season_d.permute(0, 2, 1), trend_d.permute(0, 2, 1), res_d.permute(0, 2, 1)
        season_w, trend_w, res_w = season_w.permute(0, 2, 1), trend_w.permute(0, 2, 1), res_w.permute(0, 2, 1)
        # Linear Layer
        if self.individual:
            season_d_output = torch.zeros([season_w.size(0), season_w.size(1), self.pred_len], dtype=season_w.dtype).to(season_w.device)
            trend_d_output = torch.zeros([trend_w.size(0), trend_w.size(1), self.pred_len], dtype=trend_w.dtype).to(trend_w.device)
            res_d_output = torch.zeros([res_w.size(0), res_w.size(1), self.pred_len], dtype=res_w.dtype).to(res_w.device)

            season_w_output = torch.zeros([season_w.size(0), season_w.size(1), self.pred_len], dtype=season_w.dtype).to(season_w.device)
            trend_w_output = torch.zeros([trend_w.size(0), trend_w.size(1), self.pred_len], dtype=trend_w.dtype).to(trend_w.device)
            res_w_output = torch.zeros([res_w.size(0), res_w.size(1), self.pred_len], dtype=res_w.dtype).to(res_w.device)
            for i in range(self.channels):
                season_d_output[:, i, :] = self.Linear_Seasonal_d[i](season_d[:, i, :])
                trend_d_output[:, i, :] = self.Linear_Trend_d[i](trend_d[:, i, :])
                res_d_output[:, i, :] = self.Linear_Res_d[i](res_d[:, i, :])

                season_w_output[:, i, :] = self.Linear_Seasonal_w[i](season_w[:, i, :])
                trend_w_output[:, i, :] = self.Linear_Trend_w[i](trend_w[:, i, :])
                res_w_output[:, i, :] = self.Linear_Res_w[i](res_w[:, i, :])

        else:
            season_d_output = self.season_d(season_d)  # B, D, pre_len
            trend_d_output = self.trend_d(trend_d)  # B, D, pre_len
            res_d_output = self.res_d(res_d)  # B, D, pre_len

            season_w_output = self.season_w(season_w)  # B, D, pre_len
            trend_w_output = self.trend_w(trend_w)  # B, D, pre_len
            res_w_output = self.res_w(res_w)  # B, D, pre_len

        x = season_d_output + trend_d_output + res_d_output + season_w_output + trend_w_output + res_w_output
        x = x.permute(0, 2, 1)

        return x  # B, P, D


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.norm = configs.norm
        self.pred_len = configs.pred_len
        if configs.multi_cycle:
            self.model = BDNet_BDS(configs)
        else:
            self.model = BDNet_BD(configs)

    def encoder(self, x):
        # x: [B, L, D]
        # Normalization
        if self.norm:
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x = x - x_mean
            x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = x / torch.sqrt(x_var)

        x = self.model(x)

        if self.norm:
            x = x * torch.sqrt(x_var) + x_mean
        return x  # [Batch, Output length, Channel]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.encoder(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
