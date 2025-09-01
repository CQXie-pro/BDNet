import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self, configs):
        super(Backbone, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.d_model = 1024
        self.model_type = configs.model_type

        if self.model_type == "linear":
            self.linear = nn.Linear(self.seq_len, self.pred_len)
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
        self.dropout = nn.Dropout(configs.dropout)

        self.Linear = nn.ModuleList([
            Backbone(configs) for _ in range(configs.enc_in)
        ]) if individual else Backbone(configs)

    def encoder(self, x):
        # x: [B, L, D]
        x = x.permute(0, 2, 1)

        # Linear Layer
        if self.individual:
            x_output = torch.zeros([x.size(0), x.size(1), self.pred_len], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                x_output[:, i, :] = self.Linear[i](x[:, i, :])
        else:
            x_output = self.Linear(x)

        return x_output.permute(0, 2, 1)  # [Batch, Output length, Channel]

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
