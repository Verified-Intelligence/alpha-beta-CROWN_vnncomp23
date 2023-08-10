from torch.nn import functional as F
from auto_LiRPA.bound_ops import (BoundLinear, BoundConv,
                                  BoundBatchNormalization, BoundAdd, BoundCast)


def compute_ratio(lower_bound, upper_bound):
    lower_temp = lower_bound.clamp(max=0)
    upper_temp = F.relu(upper_bound)
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio
    return slope_ratio, intercept


def get_preact_params(act, zero_default=False):
    # Legacy code for getting bias when there is a single input
    assert len(act.inputs) == 1
    return get_babsr_biases(act, zero_default)[0]


def get_babsr_biases(act, zero_default=False):
    """A new function for getting the bias term."""
    biases = []
    for input_node in act.inputs:
        if type(input_node) == BoundConv:
            if len(input_node.inputs) > 2:
                bias = input_node.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
            else:
                bias = 0
        elif type(input_node) == BoundLinear:
            # TODO: consider if no bias in the BoundLinear layer
            bias = input_node.inputs[2].param.detach()
        elif type(input_node) == BoundAdd:
            bias = 0
            for l in input_node.inputs:
                if type(l) == BoundConv:
                    if len(l.inputs) > 2:
                        bias += l.inputs[-1].param.detach()
                if type(l) == BoundBatchNormalization:
                    bias += 0  # TODO l.inputs[-3].param.detach()
                if type(l) == BoundAdd:
                    for ll in l.inputs:
                        # Check length to skip conv layers without bias.
                        if type(ll) == BoundConv and len(ll.inputs) > 2:
                            bias += ll.inputs[-1].param.detach()
        elif type(input_node) == BoundBatchNormalization:
            # for BN, bias is the -3th inputs
            bias = input_node.inputs[-3].param.detach()
        else:
            if zero_default:
                bias = 0
                print('Warning: no bias found for', input_node)
            else:
                raise NotImplementedError(type(input_node))
        biases.append(bias)
    return biases
