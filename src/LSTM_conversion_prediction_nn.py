# AI-related
from torch import nn
import torch.cuda.amp


class LSTMmodel(nn.Module):

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1,
                 drop_prob: float = 0.2, device: torch.device = torch.device('cpu'), stateful: bool = False):
        # init nn.Module parent class
        super().__init__()
        # size of vector of hidden parameters passed between layers (parallel to layer width)
        self.hidden_size = hidden_size
        # number of input features to 0-th layer
        self.n_features = n_features
        # number of stacked layers
        self.num_layers = num_layers
        # where is the model located/computations done
        self.device = device
        # stateful persistent hidden parameters (whether to reset them on each batch)
        self.stateful = stateful
        self._hidden = None
        # init LSTM layer. Batch first defines format of input data as (batch_size, sequence length, input size)
        self.LSTM = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=drop_prob)
        # reduce LSTM output size (hidden_size, projection = 0) to single float prediction of CVR
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        # easy way of getting batch_size
        batch_size = x.sorted_indices.shape[0]
        # in case of stateless or forgotten init (still _hidden = None)
        if not self.stateful or not self._hidden:
            self._hidden = self.init_hidden(batch_size)
        # call forward pass of LSTM, store (hn, cn)
        _, self._hidden = self.LSTM(x, self._hidden)
        self.detach_hidden()
        # take only last (trained) prediction after all whole sequences were processed. _hidden[0] contains all layers, only final output is desired.
        # reduce the output to single float prediction
        out = self.fc(self._hidden[0][-1])
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        # hidden is of same type as weights, only different size.
        # size (num_layers, N, H_{out}), create directly at device for speedup.
        # not update at backpropagation, memory stored during forward passes.
        hidden = (weight.new_zeros(self.num_layers, batch_size, self.hidden_size, device=self.device, requires_grad=False),
                  weight.new_zeros(self.num_layers, batch_size, self.hidden_size, device=self.device, requires_grad=False))
        return hidden

    def change_device(self, device):
        self.device = device
        self.to(device)

    def detach_hidden(self) -> None:
        self._hidden[0].detach_()
        self._hidden[1].detach_()
