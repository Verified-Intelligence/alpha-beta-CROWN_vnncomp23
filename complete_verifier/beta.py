import arguments

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet


def set_beta(self: 'LiRPANet', d, beta, batch, bias=True):
    if not beta:
        for m in self.net.splittable_activations:
            m.beta = None
        return

    # count how many split nodes in each batch example (batch, num of layers)
    splits_per_example = []
    max_splits_per_layer = {}
    for bi in range(batch):
        splits_per_example.append({})
        for k, v in d['history'][bi].items():
            # First element of layer_splits is a list of split neuron IDs.
            splits_per_example[bi][k] = len(v[0])
            max_splits_per_layer[k] = max(
                max_splits_per_layer.get(k, 0), splits_per_example[bi][k])

    # Create and load warmup beta.
    self.reset_beta(batch, max_splits_per_layer, betas=d['betas'], bias=bias)

    for node in self.split_nodes:
        if node.sparse_betas is None:
            continue
        sparse_betas = (node.sparse_betas
                        if isinstance(node.sparse_betas, list)
                        else node.sparse_betas.values())
        for sparse_beta in sparse_betas:
            sparse_beta.apply_splits(d['history'], node.name)

    return splits_per_example


def reset_beta(self: 'LiRPANet', batch, max_splits_per_layer, betas=None,
               bias=False):
    for layer_name in max_splits_per_layer:
        layer = self.net[layer_name]
        start_nodes = []
        for act in self.split_activations[layer_name]:
            start_nodes.extend(list(act[0].alpha.keys()))
        shape = (batch, max_splits_per_layer[layer_name])
        if betas is not None and betas[0] is not None and layer_name in betas[0]:
            betas_ = [(betas[bi][layer_name] if betas[bi] is not None else None)
                      for bi in range(batch)]
        else:
            betas_ = [None for _ in range(batch)]
        self.net.reset_beta(layer, shape, betas_, bias=bias,
                            start_nodes=list(set(start_nodes)))


def get_beta(self: 'LiRPANet', splits_per_example, device=None):
    # split_per_example only has half of the examples.
    beta_crown_args = arguments.Config["solver"]["beta-crown"]
    enable_opt_interm_bounds = beta_crown_args['enable_opt_interm_bounds']
    ret = []
    for i in range(len(splits_per_example)):
        betas = {}
        for k in splits_per_example[i]:
            if not enable_opt_interm_bounds:
                betas[k] = self._transfer(
                    self.net[k].sparse_betas[0].val[i, :splits_per_example[i][k]], device)
            else:
                betas[k] = []
                for sparse_beta in self.net[k].sparse_betas.values():
                    betas[k].append(self._transfer(
                        sparse_beta.val[i, :splits_per_example[i][k]], device))
        ret.append(betas)
    return ret
