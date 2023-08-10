import torch
import torch.nn as nn
import numpy as np


class LinearReshapeBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 512)
        self.bn = nn.BatchNorm2d(128)

    def forward(self, x):
        out = self.linear(x)
        out = out.reshape(-1, 128, 2, 2)
        out = self.bn(out)
        return out
    

class SimplifiedLinearReshapeBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 512)

    def forward(self, x):
        out = self.linear(x)
        out = out.reshape(-1, 128, 2, 2)
        return out
    

class BNReshapeLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(128)
        self.linear = nn.Linear(512, 1)

    def forward(self, x):
        out = self.bn(x)
        out = out.reshape(-1, 512)
        out = self.linear(out)
        return out
    

class SimplifiedBNReshapeLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 1)

    def forward(self, x):
        out = x.reshape(-1, 512)
        out = self.linear(out)
        return out
    

def merge_gemm_reshape_bn(model):
    old_state_dict = model.state_dict()
    old_linear_weight = old_state_dict['linear.weight'].numpy()
    old_linear_bias = old_state_dict['linear.bias'].numpy()
    old_bn_weight = old_state_dict['bn.weight'].numpy()
    old_bn_bias = old_state_dict['bn.bias'].numpy()
    old_bn_mean = old_state_dict['bn.running_mean'].numpy()
    old_bn_var = old_state_dict['bn.running_var'].numpy()

    new_model = SimplifiedLinearReshapeBN()
    new_linear_weight = old_linear_weight.reshape(128, 4, -1).transpose(2, 1, 0) * old_bn_weight / np.sqrt(old_bn_var + model.bn.eps)
    new_linear_bias = old_linear_bias.reshape(128, 4).transpose(1, 0) * old_bn_weight / np.sqrt(old_bn_var + model.bn.eps) + old_bn_bias - old_bn_weight * old_bn_mean / np.sqrt(old_bn_var + model.bn.eps)
    new_linear_weight = new_linear_weight.transpose(0, 2, 1).reshape(-1, 512).transpose(1, 0)
    new_linear_bias = new_linear_bias.transpose(1, 0).reshape(512)
    new_state_dict = {
        'linear.weight': torch.tensor(new_linear_weight),
        'linear.bias': torch.tensor(new_linear_bias)
    }
    new_model.load_state_dict(new_state_dict)
    
    return new_model


def merge_bn_reshape_gemm(model):
    old_state_dict = model.state_dict()
    old_linear_weight = old_state_dict['linear.weight'].numpy()
    old_linear_bias = old_state_dict['linear.bias'].numpy()
    old_bn_weight = old_state_dict['bn.weight'].numpy()
    old_bn_bias = old_state_dict['bn.bias'].numpy()
    old_bn_mean = old_state_dict['bn.running_mean'].numpy()
    old_bn_var = old_state_dict['bn.running_var'].numpy()
    
    new_model = SimplifiedBNReshapeLinear()
    old_bn_weight_flatten = old_bn_weight[:, None].repeat(4, axis=-1).reshape(512)
    old_bn_bias_flatten = old_bn_bias[:, None].repeat(4, axis=-1).reshape(512)
    old_bn_mean_flatten = old_bn_mean[:, None].repeat(4, axis=-1).reshape(512)
    old_bn_var_flatten = old_bn_var[:, None].repeat(4, axis=-1).reshape(512)
    
    new_linear_weight = old_linear_weight * old_bn_weight_flatten / np.sqrt(old_bn_var_flatten + model.bn.eps)
    new_linear_bias = np.matmul(old_linear_weight, (old_bn_bias_flatten - old_bn_weight_flatten * old_bn_mean_flatten / np.sqrt(old_bn_var_flatten + model.bn.eps))) + old_linear_bias
    
    new_state_dict = {
        'linear.weight': torch.tensor(new_linear_weight),
        'linear.bias': torch.tensor(new_linear_bias)
    }
    new_model.load_state_dict(new_state_dict)
    
    return new_model
    


if __name__ == '__main__':
    torch.random.manual_seed(0)

    with torch.no_grad():
        model = LinearReshapeBN()
        model.eval()
        diff = 0
        for i in range(100):
            x = torch.randn(1, 5)
            out = model(x)
            new_model = merge_gemm_reshape_bn(model)
            new_out = new_model(x)
            # print('Original:')
            # print(out[0, 0])
            # print()
            # print('Simplified:')
            # print(new_out[0, 0])
            # input()

            diff += (new_out - out).sum().abs()
        if diff / 100 < 1e-5:
            print('Success! Error: {}'.format(diff/100))
        else:
            print('Failure! Error: {}'.format(diff/100))

    with torch.no_grad():
        model = BNReshapeLinear()
        model.eval()
        diff = 0
        for i in range(100):
            x = torch.randn(1, 128, 2, 2)
            out = model(x)
            new_model = merge_bn_reshape_gemm(model)
            new_out = new_model(x)
            # print('Original:')
            # print(out[0, 0])
            # print()
            # print('Simplified:')
            # print(new_out[0, 0])
            # input()

            diff += (new_out - out).sum().abs()
        if diff / 100 < 1e-5:
            print('Success! Error: {}'.format(diff/100))
        else:
            print('Failure! Error: {}'.format(diff/100))

    
