import torch
import arguments

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet


def init_alpha_mask(lb, all_invalid=False):
    use_alpha_mask = arguments.Config['solver']['beta-crown']['alpha_masks']
    first_lb = next(iter(lb.values()))
    batch_size = first_lb.shape[0]
    if use_alpha_mask:
        return (torch.ones(batch_size, dtype=torch.bool,
                           device=first_lb.device) if all_invalid
                else torch.zeros(batch_size, dtype=torch.bool,
                           device=first_lb.device))
    else:
        return None

    # use_alpha_mask = arguments.Config['solver']['beta-crown']['alpha_masks']
    # if use_alpha_mask:
    #     alpha_masks = {k: torch.ones_like(v).to(torch.bool) if all_invalid
    #                    else torch.zeros_like(v).to(torch.bool)
    #                    for k, v in lb.items()}
    # else:
    #     alpha_masks = {}
    # return alpha_masks


def copy_alpha(self: 'LiRPANet', reference_alphas, num_targets,
               target_batch_size=None, now_batch=None, interm_bounds=None,
               batch_size=None):
    # alpha manipulation, since after init_alpha all things are copied
    # from alpha-CROWN and these alphas may have wrong shape
    opt_interm_bounds = (
        arguments.Config['solver']['beta-crown']['enable_opt_interm_bounds']
        or arguments.Config['bab']['branching']['new_input_split']['enable'])
    final_name = self.net.final_node().name
    for m in self.net.optimizable_activations:
        keys = list(m.alpha.keys())
        # when fixed intermediate bounds are available, since intermediate betas
        # are not used anymore because we use fixed intermediate bounds later,
        # we can delete these intermediate betas to save space
        if interm_bounds is not None and not opt_interm_bounds:
            for k in keys:
                if k not in self.alpha_start_nodes:
                    del m.alpha[k]
        if final_name not in m.alpha:
            continue
        if (m.alpha[final_name].shape[1] != 1
                or m.alpha[final_name].shape[2] != batch_size):
            # shape mismatch detected
            # pick the first slice with shape [2, 1, 1, ...],
            # and repeat to [2, 1, batch_size, ...]
            repeat = [1 if i != 2 else batch_size
                    for i in range(m.alpha[final_name].dim())]
            m.alpha[final_name] = m.alpha[final_name][:, 0:1, 0:1].repeat(*repeat)

    if reference_alphas is None:
        return False

    # We already have alphas available
    all_alpha_initialized = True
    for m in self.net.optimizable_activations:
        for spec_name, alpha in m.alpha.items():
            # each alpha size is (2, spec, batch_size, *shape); batch size is 1.
            if not spec_name in reference_alphas[m.name]:
                continue
            reference_alpha = reference_alphas[m.name][spec_name]
            if spec_name == self.net.final_node().name:
                target_start = now_batch * target_batch_size
                target_end = min((now_batch + 1) * target_batch_size, num_targets)
                if alpha.size()[2] == target_end - target_start:
                    print(f'setting alpha for layer {m.name} '
                          f'start_node {spec_name} with alignment adjustment')
                    # The reference alpha has deleted the pred class itself,
                    # while our alpha keeps that
                    # now align these two
                    # note: this part actually implements the following
                    # TODO (extract alpha according to different label)
                    if reference_alpha.size()[1] > 1:
                        # didn't apply multiple x in incomplete_verifier
                        alpha.data = reference_alpha[:, target_start:target_end].reshape_as(alpha.data)
                    else:
                        # applied multiple x in incomplete_verifier
                        alpha.data = reference_alpha[:, :, target_start:target_end].reshape_as(alpha.data)
                else:
                    all_alpha_initialized = False
                    print(f'not setting layer {m.name} start_node {spec_name} '
                          'because shape mismatch '
                          f'({alpha.size()} != {reference_alpha.size()})')
            elif alpha.size() == reference_alpha.size():
                print(f"setting alpha for layer {m.name} start_node {spec_name}")
                alpha.data.copy_(reference_alpha)
            elif all([si == sj or ((d == 2) and sj == 1)
                      for d, (si, sj) in enumerate(
                          zip(alpha.size(), reference_alpha.size()))]):
                print(f'setting alpha for layer {m.name} start_node {spec_name} '
                      'with batch sample broadcasting')
                alpha.data.copy_(reference_alpha)
            else:
                # TODO extract alpha according to different label
                all_alpha_initialized = False
                print(f'not setting layer {m.name} start_node {spec_name} '
                      'because shape mismatch '
                      f'({alpha.size()} != {reference_alpha.size()})')

    return all_alpha_initialized


def get_alpha(self: 'LiRPANet', only_final=False, half=False, device=None):
    # alpha has size (2, spec, batch, *shape). When we save it,
    # we make batch dimension the first.
    # spec is some intermediate layer neurons, or output spec size.
    new_input_split = arguments.Config['bab']['branching']['new_input_split']['enable']
    if new_input_split:
        only_final = False
    ret = {}
    for m in self.net.perturbed_optimizable_activations:
        ret[m.name] = {}
        for spec_name, alpha in m.alpha.items():
            if not only_final or spec_name in self.alpha_start_nodes:
                ret[m.name][spec_name] = self._transfer(alpha, device, half=half)
    return ret


def set_alpha(self: 'LiRPANet', d, set_all=False):
    new_input_split = arguments.Config['bab']['branching']['new_input_split']['enable']
    use_alpha_masks = arguments.Config['solver']['beta-crown']['alpha_masks']
    if new_input_split:
        set_all = True

    alpha = d['alphas']
    if len(alpha) == 0:
        return

    for m in self.net.perturbed_optimizable_activations:
        for spec_name in list(m.alpha.keys()):
            if spec_name in alpha[m.name]:
                # Only setup the last layer alphas if no refinement is done.
                if spec_name in self.alpha_start_nodes or set_all:
                    m.alpha[spec_name] = alpha[m.name][spec_name]
                    # Duplicate for the second half of the batch.
                    m.alpha[spec_name] = m.alpha[spec_name].detach().requires_grad_()
            else:
                # This layer's alpha is not used.
                # For example, we can drop all intermediate layer alphas.
                del m.alpha[spec_name]

    if 'alpha_mask_domains' in d:
        for node in self.net.nodes():
            node.alpha_mask_domains = d['alpha_mask_domains']

    # if use_alpha_masks:
    #     if 'alpha_masks' in d:
    #         alpha_masks = d['alpha_masks']
    #         for node in self.net.nodes():
    #             node.alpha_mask_domains = alpha_masks
    #         # for k, v in alpha_masks.items():
    #         #     self.net[k].alpha_mask_domains = v
    #     else:
    #         for node in self.net.nodes():
    #             node.alpha_mask_domains = None
