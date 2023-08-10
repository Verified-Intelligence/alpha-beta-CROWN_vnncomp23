"""
Old branching heuristics, must be removed very soon (assigned to Kaidi).
"""

import math
import torch
import arguments


@torch.no_grad()
def input_split_branching(net, dom_lb, x_L, x_U, lA, thresholds,
                          branching_method, split_depth=1, num_iter=0, cs=None):
    """
    Produce input split according to branching methods.
    """
    x_L = x_L.flatten(1)
    x_U = x_U.flatten(1)

    if branching_method == 'naive':
        # we just select the longest edge
        return torch.topk(x_U - x_L, split_depth, -1).indices
    elif branching_method == 'sb':
        branching_args = arguments.Config['bab']['branching']
        input_split_args = branching_args['input_split']
        lA_clamping_thresh = branching_args['sb_coeff_thresh']
        sb_margin_weight = input_split_args['sb_margin_weight']
        sb_primary_spec = input_split_args['sb_primary_spec']
        sb_primary_spec_iter = input_split_args['sb_primary_spec_iter']
        sb_sum = input_split_args['sb_sum']

        lA = lA.view(lA.shape[0], lA.shape[1], -1)
        # lA shape: (batch, spec, # inputs)
        perturb = (x_U - x_L).unsqueeze(-2)
        # perturb shape: (batch, 1, # inputs)
        # dom_lb shape: (batch, spec)
        # thresholds shape: (batch, spec)
        assert lA_clamping_thresh >= 0

        if (sb_primary_spec is not None
                and num_iter is not None and num_iter >= sb_primary_spec_iter):
            score = (lA[:, sb_primary_spec].abs().clamp(min=lA_clamping_thresh)
                     * perturb.squeeze(1) / 2
                    + (dom_lb[:, sb_primary_spec].to(lA.device).unsqueeze(-1)
                        - thresholds[:, sb_primary_spec].unsqueeze(-1))
                    * sb_margin_weight)
        else:
            score = (lA.abs().clamp(min=lA_clamping_thresh) * perturb / 2
                    + (dom_lb.to(lA.device).unsqueeze(-1)
                        - thresholds.unsqueeze(-1)) * sb_margin_weight)
            if sb_sum:
                score = score.sum(dim=-2)
            else:
                score = score.amax(dim=-2)
        # note: the k (split_depth) in topk <= # inputs, because split_depth is computed as
        # min(max split depth, # inputs).
        # 1) If max split depth <= # inputs, then split_depth <= # inputs.
        # 2) If max split depth > # inputs, then split_depth = # inputs.
        return torch.topk(score, split_depth, -1).indices
    elif branching_method == 'brute-force':
        assert x_L.ndim == 2
        input_dim = x_L.shape[1]
        x_M = (x_L + x_U) / 2
        new_x_L = x_L.expand(2, input_dim, -1, -1).clone()
        new_x_U = x_U.expand(2, input_dim, -1, -1).clone()
        for i in range(input_dim):
            new_x_U[0, i, :, i] = x_M[:, i]
            new_x_L[1, i, :, i] = x_M[:, i]
        new_x_L = new_x_L.view(-1, new_x_L.shape[-1])
        new_x_U = new_x_U.view(-1, new_x_U.shape[-1])
        from auto_LiRPA import BoundedTensor, PerturbationLpNorm
        new_x = BoundedTensor(
            new_x_L,
            ptb=PerturbationLpNorm(x_L=new_x_L, x_U=new_x_U))
        C = net.c.expand(new_x.shape[0], -1, -1)
        lb_ibp = net.net.compute_bounds(
            x=(new_x,), C=C, method='ibp', bound_upper=False, return_A=False)[0]
        # reference_interm_bounds = {}
        # for node in net.net.nodes():
        #     if (node.perturbed
        #         and isinstance(getattr(node, 'lower', None), torch.Tensor)
        #         and isinstance(getattr(node, 'upper', None), torch.Tensor)):
        #         reference_interm_bounds[node.name] = (node.lower, node.upper)
        # lb_crown = net.net.compute_bounds(
        #     x=(new_x,), C=C, method='crown', bound_upper=False,
        #     reference_bounds=reference_interm_bounds
        # )[0]
        # lb = torch.max(lb_ibp, lb_crown)
        lb = lb_ibp
        objective = (lb - thresholds[0]).view(2, input_dim, -1, lb.shape[-1])
        objective = objective.amax(-1).amin(dim=0)
        return objective.argmax(0).unsqueeze(-1)
    else:
        raise NameError(f'Unsupported branching method "{branching_method}" for input splits.')


@torch.no_grad()
def input_split_parallel(x_L, x_U, shape=None,
                         cs=None, thresholds=None, split_depth=1, i_idx=None, split_partitions=2):
    """
    Split the x_L and x_U given split_idx and split_depth.
    """
    # FIXME: this function should not be in this file.
    x_L = x_L.flatten(1)
    x_U = x_U.flatten(1)

    x_L_cp = x_L.clone()
    x_U_cp = x_U.clone()

    split_depth = min(split_depth, i_idx.size(1))
    remaining_depth = split_depth
    input_dim = x_L.shape[1]
    while remaining_depth > 0:
        for i in range(min(input_dim, remaining_depth)):
            indices = torch.arange(x_L_cp.shape[0])
            copy_num = x_L_cp.shape[0]//x_L.shape[0]
            idx = i_idx[:,i].repeat(copy_num).long()

            x_L_cp_list, x_U_cp_list = [], []
            for partition in range(split_partitions):
                x_L_cp_tmp = x_L_cp.clone()
                x_U_cp_tmp = x_U_cp.clone()

                lrange = ((partition + 1) * x_L_cp[indices, idx] +
                          (split_partitions - partition - 1) * x_U_cp[indices, idx]) / split_partitions
                urange = (partition * x_L_cp[indices, idx] +
                          (split_partitions - partition) * x_U_cp[indices, idx]) / split_partitions

                x_L_cp_tmp[indices, idx] = lrange
                x_U_cp_tmp[indices, idx] = urange

                x_L_cp_list.append(x_L_cp_tmp)
                x_U_cp_list.append(x_U_cp_tmp)

            x_L_cp = torch.cat(x_L_cp_list)
            x_U_cp = torch.cat(x_U_cp_list)

            # x_L_cp_tmp = x_L_cp.clone()
            # x_U_cp_tmp = x_U_cp.clone()
            #
            # mid = (x_L_cp[indices, idx] + x_U_cp[indices, idx]) / 2
            #
            # x_L_cp[indices, idx] = mid
            # x_U_cp_tmp[indices, idx] = mid
            # x_L_cp = torch.cat([x_L_cp, x_L_cp_tmp])
            # x_U_cp = torch.cat([x_U_cp, x_U_cp_tmp])
        remaining_depth -= min(input_dim, remaining_depth)

    split_depth = split_depth - remaining_depth

    new_x_L = x_L_cp.reshape(-1, *shape[1:])
    new_x_U = x_U_cp.reshape(-1, *shape[1:])

    if cs is not None:
        cs_shape = [split_partitions ** split_depth] + [1] * (len(cs.shape) - 1)
        cs = cs.repeat(*cs_shape)
    if thresholds is not None:
        thresholds = thresholds.repeat(split_partitions ** split_depth, 1)
    return new_x_L, new_x_U, cs, thresholds, split_depth


def get_split_depth(x_L, split_partitions=2):
    split_depth = 1
    if len(x_L) < arguments.Config["solver"]["min_batch_size_ratio"] * arguments.Config["solver"]["batch_size"]:
        min_batch_size = arguments.Config["solver"]["min_batch_size_ratio"] * arguments.Config["solver"]["batch_size"]
        split_depth = int(math.log(min_batch_size//len(x_L))//math.log(split_partitions))
        split_depth = max(split_depth, 1)
    return split_depth
