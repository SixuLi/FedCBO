import torch
import torch.nn.functional as F
import numpy as np

from torch import nn


class FCModel(nn.Module):
    """
    FC-NN model.
    The model has n hidden layer each consisting of linear network followed by
    ReLU activations as non-linearity.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, bias=False):
        """
        input_dim: The input dimension
        hidden_dims: List of hidden dimensions for this model
        output_dim: Output dimension
        bias: If the linear elements should have bias
        """
        super(FCModel, self).__init__()
        self.bias = bias
        self.channels = [input_dim] + hidden_dims + [output_dim]
        self.layers = []

        for idx in range(1, len(self.channels)):
            cur_layer = [nn.Linear(self.channels[idx-1], self.channels[idx], bias=self.bias)]
            if idx + 1 < len(self.channels):
                cur_layer.append(nn.ReLU())
            seq_cur_layer = nn.Sequential(*cur_layer)
            self.layers.append(seq_cur_layer)
        self.layers_aggregated = nn.Sequential(*self.layers)

    def get_model_config(self):
        return {'input_dim': self.channels[0],
                'hidden_dims': self.channels[1:-1],
                'output_dim': self.channels[-1],
                'is_bias': self.bias}

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers_aggregated(x)

    @property
    def num_layers(self):
        return len(self.layers)

    @property
    def input_dim(self):
        return self.channels[0]


    def get_layer_weights(self, layer_num=1):
        assert 0 < layer_num <= self.num_layers
        # Returns the weights from the linear layer of this model
        return self.layers[layer_num - 1]._modules['0'].weight

    def get_layer_bias(self, layer_num=1):
        assert 0 < layer_num <= self.num_layers
        # Returns the bias from the linear layer of this model
        return self.layers[layer_num - 1]._modules['0'].bias



def test_fcnn_model():
    model = FCModel(input_dim=728, hidden_dims=[100], output_dim=10, bias=True)
    print(model)
    print(model.get_layer_bias(layer_num=1))
    batch_size = 16
    data = torch.rand(batch_size, 728)
    output = model(data)
    assert output.size(0) == batch_size
    assert output.size(1) == 10
    print('FCNN works!')


if __name__ == '__main__':
    test_fcnn_model()