import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.individual = individual
        self.channels = configs.enc_in
        self.dropout = nn.Dropout(configs.dropout)

        self.Linear = nn.ModuleList([
            nn.Linear(configs.seq_len, configs.pred_len) for _ in range(configs.enc_in)
        ]) if individual else nn.Linear(configs.seq_len, configs.pred_len)

    def encoder(self, x):
        # x: [B, L, D]
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        x = self.dropout(x)
        x = x.permute(0, 2, 1)

        # Linear Layer
        if self.individual:
            x_output = torch.zeros([x.size(0), x.size(1), self.pred_len], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                x_output[:, i, :] = self.Linear[i](x[:, i, :])
        else:
            x_output = self.Linear(x)

        # iRIN
        x_output = x_output.permute(0, 2, 1) * torch.sqrt(x_var) + x_mean
        return x_output

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
