import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sigma = 0.0, return_hidden = False):
        super().__init__()
        self.input_size, self.hidden_size, self.output_size, self.sigma = input_size, hidden_size, output_size, sigma
        self.w_in = nn.Linear(input_size, hidden_size, bias=False)
        self.w = nn.Linear(hidden_size, hidden_size, bias=True)
        self.w_out = nn.Linear(hidden_size, output_size, bias=True)
        self.f = nn.ReLU()
        self.f_out = nn.Softmax(dim=-1)
        self.return_hidden = return_hidden
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.w.weight)
        nn.init.xavier_uniform_(self.w_in.weight)
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.zeros_(self.w.bias)
        nn.init.zeros_(self.w_out.bias)

    def forward(self, x, h0=None):
        batch, time, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype) if h0 is None else h0
        _h = []
        for t in range(time):
            noise = self.sigma * torch.randn_like(h)
            if self.sigma != 0:
                noise = torch.normal(0, self.sigma, size=h.shape, device=h.device, dtype=h.dtype)
            h = self.f(self.w_in(x[:, t]) + self.w(h) + noise)
            if self.return_hidden:
                _h.append(h)
        return (self.f_out(self.w_out(h)), torch.stack(_h, dim = 1)) if self.return_hidden else self.f_out(self.w_out(h))