#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
## Copyright (C) 2021-2022, Huan Zhang <huan@huan-zhang.com>           ##
##                     Kaidi Xu, Zhouxing Shi, Shiqi Wang              ##
##                     Linyi Li, Jinqi (Kathryn) Chen                  ##
##                     Zhuolin Yang, Yihan Wang                        ##
##                                                                     ##
##      See CONTRIBUTORS for author contacts and affiliations.         ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import os
import copy
from collections import defaultdict

import torch
import arguments
import warnings

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import stop_criterion_placeholder, stop_criterion_all

from attack import attack_after_crown
from attack.attack_pgd import save_cex, process_vnn_lib_attack
from input_split.input_split_on_relu_domains import input_branching_decisions
from utils import Timer
from load_model import Customized
from prune import PruneAfterCROWN, prune_alphas
from domain_updater import (DomainUpdater, DomainUpdaterSimple,
                            DomainUpdaterInputSplit)
from cuts.implied_cuts import DomainUpdaterImpliedCuts
from heuristics.nonlinear import precompute_A
### preprocessor-hint: private-section-start
from cuts.optimized_cuts import *
### preprocessor-hint: private-section-end


class LiRPANet:
    def __init__(self, model_ori, in_size, c=None, device=None,
                 cplex_processes=None):
        """
        convert pytorch model to auto_LiRPA module
        """
        general_args = arguments.Config['general']
        solver_args = arguments.Config['solver']
        bab_args = arguments.Config['bab']
        self.c = c
        self.model_ori = model_ori
        self.input_shape = in_size
        self.device = device or general_args['device']
        net = copy.deepcopy(model_ori)
        self.net = BoundedModule(
            net, torch.zeros(in_size, device=self.device),
            bound_opts={
                'deterministic': general_args['deterministic'],
                'conv_mode': general_args['conv_mode'],
                'sparse_features_alpha': general_args['sparse_alpha'],
                'sparse_spec_alpha': general_args['sparse_alpha'],
                'sparse_intermediate_bounds': general_args['sparse_interm'],
                'crown_batch_size': solver_args['crown']['batch_size'],
                'max_crown_size': solver_args['crown']['max_crown_size'],
                'forward_refinement': solver_args['forward']['refine'],
                'dynamic_forward': solver_args['forward']['dynamic'],
                'forward_max_dim': solver_args['forward']['max_dim'],
                'use_full_conv_alpha': solver_args['alpha-crown']['full_conv_alpha'],
                'disable_optimization': solver_args['alpha-crown']['disable_optimization'],
                'fixed_reducemax_index': True,
                'matmul': {'share_alphas': solver_args['alpha-crown']['matmul_share_alphas']},
                'tanh': {'loose_threshold': bab_args['branching']['nonlinear_split']['loose_tanh_threshold']},
                'buffers': {'no_batchdim': general_args['no_batchdim_buffers']},
                'optimize_bound_args': {
                    'include_gammas_in_linear_layer': solver_args['alpha-crown']['include_output_constraint'],
                    'include_output_constraint': solver_args['alpha-crown']['include_output_constraint'],
                },
            },
            device=self.device
        )
        self.net = eval(general_args['graph_optimizer'])(self.net)
        self.new_input_split = bab_args['branching']['new_input_split']['enable']
        self.root = self.net[self.net.root_names[0]]

        self.net.eval()
        self.return_A = False
        self.needed_A_dict = None
        self.cutter = None # class for generating and optimizing cuts
        self.timer = Timer()

        # for fetching cplex in parallel
        self.mip_building_proc = self.processes = None
        self.cplex_processes = cplex_processes
        self.pool = self.pool_result = self.pool_termination_flag = None # For multi-process.

        # for recording whether we need to return intermediate bounds
        # after initial intermediate bounds fetching, this switch will be
        # aligned with arg.bab.interm_transfer
        self.interm_transfer = True

        # check conversion correctness
        dummy = torch.randn(in_size, device=self.device)
        try:
            assert torch.allclose(net(dummy), self.net(dummy))
        except AssertionError:
            print('torch allclose failed: norm '
                  f'{torch.norm(net(dummy) - self.net(dummy))}')

        self.final_name = self.net.final_name
        self.alpha_start_nodes = [self.final_name] + list(filter(
            lambda x: len(x.strip()) > 0,
            bab_args['optimized_interm'].split(',')))

    @torch.no_grad()
    def get_primal_upper_bound(self, A):
        assert self.x.ptb.norm == torch.inf, (
            'We only support to get primals for Linf norm perturbation so far')
        input_A_lower = A[self.net.output_name[0]][self.net.input_name[0]]['lA']
        batch = input_A_lower.shape[0]

        x_lb, x_ub = self.x.ptb.x_L, self.x.ptb.x_U
        x_lb = x_lb.repeat(batch, 1, 1, 1)
        x_ub = x_ub.repeat(batch, 1, 1, 1)
        input_primal = x_lb.clone().detach()
        input_primal[input_A_lower.squeeze(1) < 0] = x_ub[input_A_lower.squeeze(1) < 0]

        return input_primal, self.model_ori(input_primal).matmul(self.c[0].transpose(-1, -2))

    def get_interm_bounds(self, lb, ub=None, init=True, device=None):
        """Get the intermediate bounds.

        By default, we also add final layer bound after applying C
        (lb and lb+inf).
        """

        lower_bounds, upper_bounds = {}, {}
        if init:
            self._get_split_nodes()
            for layer in self.net.layers_requiring_bounds + self.split_nodes:
                lower_bounds[layer.name] = layer.lower.detach()
                upper_bounds[layer.name] = layer.upper.detach()
        elif self.interm_transfer:
            for layer in self.net.layers_requiring_bounds:
                lower_bounds[layer.name] = self._transfer(
                    layer.lower.detach(), device)
                upper_bounds[layer.name] = self._transfer(
                    layer.upper.detach(), device)

        lower_bounds[self.final_name] = lb.flatten(1).detach()
        if ub is None:
            ub = lb + torch.inf
        upper_bounds[self.final_name] = ub.flatten(1).detach()

        if self.new_input_split:
            lower_bounds[self.root.name] = self.root.lower.detach()
            upper_bounds[self.root.name] = self.root.upper.detach()

        return lower_bounds, upper_bounds

    def get_mask(self):
        masks = {}
        for node in self.net.get_splittable_activations():
            mask = []
            for idx in node.requires_input_bounds:
                input_node = node.inputs[idx]
                if not input_node.perturbed:
                    mask.append(None)
                else:
                    mask.append(node.get_split_mask(
                        input_node.lower, input_node.upper, idx))
            masks[node.name] = mask
        return masks

    def get_lA(self, preserve_mask=None, tot_cells=None,
               transpose=True, device=None):
        lAs = {}
        nodes = list(self.net.get_splittable_activations())
        if self.new_input_split:
            nodes.append(self.root)
        for node in nodes:
            lA = getattr(node, 'lA', None)
            if lA is None:
                continue
            if preserve_mask is not None:
                new_lA = torch.zeros(
                    [tot_cells, lA.shape[0]] + list(lA.shape[2:]),
                    dtype=lA.dtype, device=lA.device)
                new_lA[preserve_mask] = lA.transpose(0,1)
                lA = new_lA
            else:
                lA = lA.transpose(0, 1) if transpose else lA.squeeze(0)
            lAs[node.name] = self._transfer(lA, device)
        return lAs

    def get_candidate_parallel(self, lb, ub, device=None):
        """Get the intermediate bounds in the current model."""
        return self.get_interm_bounds(lb, ub, init=False, device=device)

    def expand_batch(self, x, batch):
        return x[0:1].expand(batch, *[-1]*(x.ndim-1))

    def expand_x(self, batch, x_L=None, x_U=None, lb=None, ub=None):
        if self.new_input_split and lb is not None:
            x_L, x_U = lb[self.root.name], ub[self.root.name]
        if x_L is None and x_U is None:
            ptb = PerturbationLpNorm(
                norm=self.x.ptb.norm, eps=self.x.ptb.eps,
                x_L=self.expand_batch(self.x.ptb.x_L, batch),
                x_U=self.expand_batch(self.x.ptb.x_U, batch))
        else:
            ptb = PerturbationLpNorm(norm=self.x.ptb.norm, x_L=x_L, x_U=x_U)
        new_x = BoundedTensor(self.expand_batch(self.x.data, batch), ptb)
        return new_x

    @torch.no_grad()
    def _expand_tensors(self, d, batch):
        lb, ub = d['lower_bounds'], d['upper_bounds']
        cs, x_Ls, x_Us = d.get('cs', None), d.get('x_Ls', None), d.get('x_Us', None)
        # Only the last element is used later.
        lb_last, ub_last = lb[self.final_name], ub[self.final_name]
        interm_bounds = {k: [lb[k], ub[k]] for k in lb if k != self.final_name}
        # create new_x here since batch may change
        new_x = self.expand_x(batch, x_Ls, x_Us, lb, ub)
        if cs is None:
            cs = None if self.c is None else self.c.expand(new_x.shape[0], -1, -1)
        return interm_bounds, lb_last, ub_last, cs, new_x, x_Ls, x_Us

    def update_bounds(self, d, beta=None, fix_interm_bounds=True,
                      shortcut=False, stop_criterion_func=stop_criterion_placeholder(),
                      multi_spec_keep_func=None, beta_bias=True):
        """Main function for computing bounds after branch and bound in Beta-CROWN."""
        deterministic_opt = arguments.Config['general']['deterministic_opt']
        solver_args = arguments.Config['solver']
        beta_args = solver_args['beta-crown']
        bab_args = arguments.Config['bab']
        if beta is None:
            # might need to set beta False in FSB node selection
            beta = beta_args['beta']
        iteration = beta_args['iteration']
        get_upper_bound = bab_args['get_upper_bound']
        enable_opt_interm_bounds = beta_args['enable_opt_interm_bounds']
        branching_input_and_activation = bab_args['branching']['branching_input_and_activation']
        batch = d['upper_bounds'][self.final_name].shape[0]
        decision_thresh = d.get('thresholds', None)

        if getattr(self, 'new_input_split_now', False):
            # Disable beta in input splits
            beta = False

        self.timer.start('func')
        self.timer.start('prepare')

        if self.net.cut_used:
            self.disable_cut_for_branching()
        if beta:
            splits_per_example = self.set_beta(d, beta, batch, bias=beta_bias)
            self.net.cut_used = (
                arguments.Config['bab']['cut']['enabled']
                and arguments.Config['bab']['cut']['bab_cut']
                and getattr(self.net, 'cut_module', None) is not None)
            # even we need to use cut, maybe the cut is not fetched yet
            if self.net.cut_used:
                iteration = self.set_cut_params(
                    batch, batch, d.get('split_history', None))
            # here to handle the case where the split node happen to be in the
            # cut constraint !!!
        ret = self._expand_tensors(d, batch)
        interm_bounds, lb_last, ub_last, c, new_x, x_Ls, x_Us = ret
        self.set_alpha(d, set_all=enable_opt_interm_bounds)
        self.timer.add('prepare')

        self.timer.start('bound')
        self.net.set_bound_opts({
            'optimize_bound_args': {
                'enable_beta_crown': beta,
                'fix_interm_bounds': fix_interm_bounds,
                'stop_criterion_func': stop_criterion_func,
                'multi_spec_keep_func': multi_spec_keep_func,
                'iteration': iteration,
            },
            'enable_opt_interm_bounds': enable_opt_interm_bounds,
        })
        self.set_crown_bound_opts('beta')

        if shortcut:
            args_compute_bounds = dict(x=(new_x,), C=c, reuse_alpha=True,
                interm_bounds=interm_bounds, bound_upper=False,
                decision_thresh=decision_thresh)
            with (torch.enable_grad() if beta else torch.no_grad()):
                lb, _, = self.net.compute_bounds(
                    method='CROWN-optimized' if beta else 'backward',
                    **args_compute_bounds)
            return lb

        # we need A matrix to construct adv example
        return_A = get_upper_bound or branching_input_and_activation
        original_size = new_x.shape[0]

        if fix_interm_bounds:
            reference_bounds = {}
            for name in self.alpha_start_nodes:
                if name in interm_bounds:
                    reference_bounds[name] = interm_bounds[name]
                    interm_bounds.pop(name)
        else:
            reference_bounds = interm_bounds
            interm_bounds = {}
        if len(reference_bounds):
            print('Recompute intermediate bounds for nodes:',
                ', '.join(list(reference_bounds.keys())))

        if getattr(self, 'new_input_split_now', False):
            method = solver_args['bound_prop_method']
        else:
            method = 'CROWN-Optimized'
        tmp_ret = self.net.compute_bounds(
            x=(new_x,), C=c, method=method,
            interm_bounds=interm_bounds, reference_bounds=reference_bounds,
            return_A=return_A, needed_A_dict=self.needed_A_dict,
            cutter=self.cutter, bound_upper=False,
            decision_thresh=decision_thresh)
        self.timer.add('bound')

        if return_A:
            lb, _, A = tmp_ret
        else:
            lb, _ = tmp_ret
        if get_upper_bound:
            primal_x, ub = self.get_primal_upper_bound(A)
        else:
            ub = torch.full_like(lb, fill_value=torch.inf)  # dummy upper bound
            primal_x = None
        # Use A matrix of the input, the find best neuron to branch in input space.
        input_split_idx = input_branching_decisions(
            self.net, lb,
            A[self.net.output_name[0]][self.net.input_name[0]]['lA'],
            x_Ls, x_Us, decision_thresh
        ) if branching_input_and_activation else None

        with torch.no_grad():
            # Move tensors to CPU for all elements in this batch.
            self.timer.start('transfer')
            lb, ub = lb.to(device='cpu'), ub.to(device='cpu')
            lAs = self.get_lA(
                self.net.last_update_preserve_mask, original_size,
                device='cpu', transpose=True)
            self.timer.add('transfer')
            self.timer.start('finalize')
            ret_s = self.get_alpha(device='cpu', half=not deterministic_opt)
            if beta:
                ret_b = self.get_beta(splits_per_example, device='cpu')
            else:
                ret_b = [{} for _ in range(batch)]
            # Reorganize tensors.
            ret_l, ret_u = self.get_candidate_parallel(lb, ub, device='cpu')
            if not deterministic_opt:
                ret_l[self.final_name] = torch.max(
                    ret_l[self.final_name], lb_last.cpu())
                if not get_upper_bound:
                    # Do not set to min so the primal is always corresponding
                    # to the upper bound.
                    ret_u[self.final_name] = torch.min(
                        ret_u[self.final_name], ub_last.cpu())
            self.timer.add('finalize')

        # Each key is corresponding to a pre-relu layer, and each value intermediate
        # beta values for neurons in that layer.
        new_split_history = [{} for _ in range(batch)]
        if self.net.cut_used:
            self.set_cut_new_split_history(new_split_history, batch)

        self.timer.add('func')
        self.timer.print()

        # Each key is corresponding to a pre-relu layer, and each value
        # intermediate beta values for neurons in that layer.,
        best_intermediate_betas = [defaultdict(dict) for _ in range(batch)]

        return {
            'lower_bounds': ret_l, 'upper_bounds': ret_u,
            'lAs': lAs, 'alphas': ret_s,
            'betas': ret_b, 'split_history': new_split_history,
            'intermediate_betas': best_intermediate_betas,
            'primals': primal_x,
            'c': c, 'x_Ls': x_Ls, 'x_Us': x_Us,
            'input_split_idx': input_split_idx,
        }

    def _get_split_nodes(self, verbose=False):
        self.net.get_split_nodes(input_split=self.new_input_split)
        self.split_nodes = self.net.split_nodes
        self.split_activations = self.net.split_activations
        if verbose:
            print('Split layers:')
            for layer in self.split_nodes:
                print(f'  {layer}: {self.split_activations[layer.name]}')
            print('Nonlinear functions:')
            for node in self.net.nodes():
                if len(node.requires_input_bounds):
                    print('  ', node)

    def build(self, input_domain, x,
              stop_criterion_func=stop_criterion_placeholder(),
              bounding_method=None, decision_thresh=None, vnnlib_ori=None):
        # TODO merge with build_with_refined_bounds()
        solver_args = arguments.Config['solver']
        bab_args = arguments.Config['bab']
        branching_args = bab_args['branching']
        share_alphas = solver_args['alpha-crown']['share_alphas']
        bounding_method = bounding_method or solver_args['bound_prop_method']
        branching_input_and_activation = branching_args['branching_input_and_activation']
        fast_A = (branching_args['method'] == 'nonlinear'
                  and branching_args['nonlinear_split']['method'] == 'shortcut')

        self.x = x
        self.input_domain = input_domain
        self.net.set_bound_opts({
            'optimize_bound_args': {'stop_criterion_func': stop_criterion_func},
            'verbosity': 1,
        })
        self.set_crown_bound_opts('alpha')
        self._get_split_nodes(verbose=True)

        # expand x to align with C's batch size for multi target verification
        batch = self.c.size()[0]
        if self.x.shape[0] == 1 and batch > 1:
            x_expand = self.expand_x(batch)
        else:
            x_expand = x

        result = {key: None for key in [
            'mask', 'lA', 'lower_bounds', 'upper_bounds',
            'alphas', 'history', 'input_split_idx', 'attack_images']}

        prune_after_crown = None
        if bounding_method == 'alpha-crown':
            # first get CROWN bounds
            # Reference bounds are intermediate layer bounds from initial CROWN bounds.
            lb, ub, aux_reference_bounds = self.net.init_alpha(
                (x_expand,), share_alphas=share_alphas, c=self.c, bound_upper=False)
            print('initial CROWN bounds:', lb, ub)

            if bab_args['cut']['enabled'] or bab_args['cut']['implication']:
                self.enable_cuts()

            if stop_criterion_func(lb).all().item():
                # Fast path. Initial CROWN bound can verify the network.
                print('Verified with initial CROWN!')
                return lb, result

            if arguments.Config['attack']['pgd_order'] == 'middle' and vnnlib_ori is not None:
                # Run adversarial attack on those specs that cannot be verified by CROWN.
                verified_success, attack_images = attack_after_crown(
                    lb, vnnlib_ori[0], self.model_ori, x, decision_thresh)
                if verified_success:  # Adversarial images are generated here.
                    print('PGD attack succeeded.')
                    if arguments.Config["general"]["save_adv_example"]:
                        try:
                            attack_output = self.model_ori(attack_images.view(-1, *attack_images.shape[1:]))
                            list_target_label_arrays, data_min_repeat, data_max_repeat = process_vnn_lib_attack(vnnlib_ori, x)
                            save_cex(attack_images, attack_output, x, vnnlib_ori, arguments.Config["attack"]["cex_path"], data_max_repeat, data_min_repeat)
                        except Exception as e:
                            print(str(e))
                            print('save adv example failed')
                    result.update({'attack_images': attack_images})
                    return lb, result

            c_to_use = self.c
            if solver_args['prune_after_crown']:
                prune_after_crown = PruneAfterCROWN(
                    self.net, self.c, lb,
                    decision_thresh=decision_thresh)
                c_to_use = prune_after_crown.c
                # This should be the only supported case for incomplete verifier.
                assert stop_criterion_func.__qualname__.split('.')[0] == 'stop_criterion_all'
                pruned_stop_criterion_func = stop_criterion_all(prune_after_crown.decision_thresh)
                self.net.set_bound_opts({
                    'optimize_bound_args': {'stop_criterion_func': pruned_stop_criterion_func},
                })

            self._set_A_options()
            ret = self.net.compute_bounds(
                x=(x_expand,), C=c_to_use, method='CROWN-Optimized',
                return_A=self.return_A, needed_A_dict=self.needed_A_dict,
                bound_upper=False, aux_reference_bounds=aux_reference_bounds,
                cutter=self.cutter)
        elif bounding_method == 'alpha-forward':
            warnings.warn('alpha-forward can only be used with input split for now')
            self.net.bound_opts['optimize_bound_args']['init_alpha'] = True
            ret = self.net.compute_bounds(
                x=(x_expand,), C=self.c, method='forward-optimized', bound_upper=False)
        else:
            with torch.no_grad():
                if bounding_method == 'init-crown':
                    assert not self.return_A
                    lb, ub, _ = self.net.init_alpha(
                        (x_expand,), share_alphas=share_alphas, c=self.c,
                        bound_upper=False)
                    ret = lb, ub
                else:
                    ret = lb, _ = self.net.compute_bounds(
                        x=(x_expand,), C=self.c, method=bounding_method,
                        bound_upper=False)

        if self.return_A:
            lb, _, A = ret
        else:
            lb, _ = ret
            A = None

        if branching_input_and_activation:
            # Use A matrix of the input, the find best neuron to branch in input space.
            input_split_idx = input_branching_decisions(
                self.net, lb,
                A[self.net.output_name[0]][self.net.input_name[0]]['lA'],
                x_expand.ptb.x_L, x_expand.ptb.x_U, decision_thresh)
        else:
            input_split_idx = None
        if prune_after_crown:
            lb = prune_after_crown.recover_lb(lb)

        print(f'initial {bounding_method} bounds:', lb)
        global_lb = lb.min().item()
        print('Worst class: (+ rhs)', global_lb)

        # DEBUG: check loose bounds
        if os.environ.get('ABCROWN_VIEW_INTERM', False):
            print('Intermediate bounds after initial alpha-CROWN:')
            self._print_interm_bounds()
            import pdb; pdb.set_trace()

        alpha = self.get_alpha()  # initial with one node only
        # for each pre-activation layer, we initial 2 lists for the two branches
        lb, ub = self.get_interm_bounds(lb)  # primals are better upper bounds
        history = self.empty_history()
        mask = self.get_mask()
        lA = self.get_lA()
        if prune_after_crown:
            lA = prune_after_crown.recover_lA_and_alpha(lA, alpha)

        if fast_A and global_lb < 0 and self.return_A:
            precompute_A(self.net, A, x_expand,
                         interm_bounds={k: (lb[k], ub[k]) for k in lb})
        if bab_args['cut']['enabled']:
            self.create_cutter(A, x_expand, lb, ub, prune_after_crown)
        if A is not None or bab_args['cut']['implication']:
            # FIXME There is not only A here. There are also biases.
            # Need to rename.
            self.A_saved = A

        result.update({
            'mask': mask, 'lA': lA, 'lower_bounds': lb, 'upper_bounds': ub,
            'alphas': alpha, 'history': history,
            'input_split_idx': input_split_idx,
        })
        return lb[self.final_name], result

    def build_with_refined_bounds(
            self, input_domain, x,
            refined_lower_bounds=None, refined_upper_bounds=None,
            activation_opt_params=None, reference_lA=None,
            reference_alphas=None, cutter=None, refined_betas=None,
            stop_criterion_func=stop_criterion_placeholder(),
            decision_thresh=None):
        solver_args = arguments.Config['solver']
        bab_args = arguments.Config['bab']
        branch_args = bab_args['branching']
        share_alphas = solver_args['alpha-crown']['share_alphas']
        target_batch_size = solver_args['multi_class']['label_batch_size']
        loss_reduction_func = arguments.Config['general']['loss_reduction_func']
        branching_input_and_activation = branch_args['branching_input_and_activation']

        self.x = x
        self.input_domain = input_domain
        self.cutter = cutter

        # expand x to align with C's batch size for multi target verification
        batch = self.c.size()[0]
        x_expand = self.expand_x(batch)
        # also, we need to expand lower and upper bounds accordingly
        if refined_lower_bounds is not None and refined_upper_bounds is not None:
            # the intermediate bounds were shared in incomplete_verifier(), we expand them here
            for k, v in refined_lower_bounds.items():
                if k != self.final_name:
                    refined_lower_bounds[k] = self.expand_batch(v, batch)
            for k, v in refined_upper_bounds.items():
                if k != self.final_name:
                    refined_upper_bounds[k] = self.expand_batch(v, batch)

        loss_reduction_func = reduction_str2func(loss_reduction_func)
        self.refined_lower_bounds = refined_lower_bounds
        self.refined_upper_bounds = refined_upper_bounds

        self._set_A_options(bab=True)

        # batch results holder
        batch_lbs, batch_ubs, lA, alphas = [], [], {}, {}
        # tot label batches
        tot_batches = (x_expand.size()[0] + target_batch_size - 1) // target_batch_size

        def _take_batch(x, idx):
            return x[idx * target_batch_size: (idx + 1) * target_batch_size]

        for now_batch in range(tot_batches):
            print(f'build_with_refined_bounds batch [{now_batch+1}/{tot_batches}]')
            torch.cuda.empty_cache()

            batch_expand = BoundedTensor(
                _take_batch(x_expand.data, now_batch),
                PerturbationLpNorm(
                    x_expand.ptb.eps, x_expand.ptb.norm,
                    _take_batch(x_expand.ptb.x_L, now_batch),
                    _take_batch(x_expand.ptb.x_U, now_batch)))
            C_batch = _take_batch(self.c, now_batch)

            # For updating which nodes are perturbed.
            self.net.set_input(batch_expand)
            if (refined_lower_bounds is not None
                    and refined_upper_bounds is not None):
                # using refined bounds with init opt crown
                interm_bounds = {
                    k: [_take_batch(refined_lower_bounds[k], now_batch),
                        _take_batch(refined_upper_bounds[k], now_batch)]
                    for k in refined_lower_bounds
                }
            else:
                interm_bounds = None

            self.net.init_alpha(
                (batch_expand,), share_alphas=share_alphas, c=C_batch,
                interm_bounds=interm_bounds,
                activation_opt_params=activation_opt_params,
                skip_bound_compute=True)

            all_alpha_initialized = self.copy_alpha(
                reference_alphas, batch_size=batch_expand.shape[0],
                num_targets=min((now_batch + 1) * target_batch_size,
                                self.c.shape[0]) - now_batch * target_batch_size,
                target_batch_size=target_batch_size, now_batch=now_batch,
                interm_bounds=interm_bounds)

            self.net.set_bound_opts({'optimize_bound_args': {
                'stop_criterion_func': stop_criterion_func,
            }})
            self.set_crown_bound_opts('alpha')

            skip_backward_pass = False
            if all_alpha_initialized and solver_args['multi_class']['skip_with_refined_bound']:
                print('all alpha initialized')
                if not self.return_A:
                    # FIXME "A" is incorrect later when calling get_lA
                    skip_backward_pass = True
                    print('directly get lb and ub from refined bounds')
                    # Make sure the shape of reference_lA looks good so that we
                    # can recover the batch_lA
                    print('c shape:', self.c.shape)
                    print('lA shapes:', [A.shape for A in reference_lA.values()])
                    # A shape: [batch, num_output, *output_shape]
                    assert all([A.shape[0] == self.c.shape[0] for A in reference_lA.values()])
                    # Try to directly recover l and u from refined_lower_bounds
                    # and refined_upper_bounds without a backward crown pass
                    # refined_lower/upper_bounds[-1]'s shape is [labels to verify, C]
                    # self.c's shape is [labels to verify, 1, C] where target labels have value -1.
                    lb = _take_batch(refined_lower_bounds[self.final_name], now_batch)
                    ub = _take_batch(refined_upper_bounds[self.final_name], now_batch)
                    ret = (lb, ub)
                else:
                    skip_backward_pass = False
                    # do a backward crown pass
                    print('true A is required, we do a full backward CROWN pass to obtain it')
                    ret = self.net.compute_bounds(
                        x=(batch_expand,), method='backward', C=C_batch,
                        return_A=self.return_A, reuse_alpha=True,
                        interm_bounds=interm_bounds,
                        needed_A_dict=self.needed_A_dict)
            else:
                print('Restore to original setting since some alphas are not '
                      'initialized yet or being asked not to skip')
                ret = self.net.compute_bounds(
                    x=(batch_expand,), method='crown-optimized',
                    return_A=self.return_A, C=C_batch,
                    interm_bounds=interm_bounds,
                    needed_A_dict=self.needed_A_dict)

            if self.return_A:
                lb, ub, A = ret
            else:
                lb, ub = ret
                A = None

            if branching_input_and_activation:
                # Use A matrix of the input, the find best neuron to branch in input space.
                input_split_idx = input_branching_decisions(
                    self.net, lb,
                    A[self.net.output_name[0]][self.net.input_name[0]]['lA'],
                    x.ptb.x_L, x.ptb.x_U, decision_thresh)
            else:
                input_split_idx = None

            print('(alpha-)CROWN with fixed intermediate bounds:', lb, ub)
            print('Intermediate layers:', ','.join(list(interm_bounds.keys())))
            batch_alphas = self.get_alpha()
            if arguments.Config['bab']['attack']['enabled']:
                # Save all alphas, which will be further refined in bab-attack.
                self.refined_alpha = reference_alphas
            if not self.new_input_split:
                # early alpha delete to save space
                # If we are not optimizing intermediate layer bounds, we do not need
                # to save all the intermediate alpha. We only keep the alpha for the
                # last layer.`
                new_batch_alphas = prune_alphas(batch_alphas, self.alpha_start_nodes)
                del batch_alphas
                batch_alphas = new_batch_alphas
            else:
                print('Keeping all the alphas for the new input split')
            for k, v in batch_alphas.items():
                if k not in alphas:
                    alphas[k] = {}
                for kk, vv in v.items():
                    if kk not in alphas[k]:
                        alphas[k][kk] = vv
                    else:
                        alphas[k][kk] = torch.cat([alphas[k][kk], vv], dim=2)

            batch_lb, batch_ub = self.get_interm_bounds(lb)  # primals are better upper bounds

            mask = self.get_mask()
            if skip_backward_pass:
                # reference_lA is already transposed back in incomplete_verifier()
                batch_lA = reference_lA
            else:
                batch_lA = self.get_lA()

            if refined_betas is not None:
                # only has batch size 1 for refined betas
                assert len(refined_betas[0]) == 1
                history, ret_b = refined_betas[0][0], refined_betas[1][0]
            else:
                history, ret_b = self.empty_history(), None

            batch_lbs.append(batch_lb)
            batch_ubs.append(batch_ub)
            for k, v in batch_lA.items():
                if k not in lA:
                    lA[k] = v
                else:
                    # Need to accumulate itemwise over the 0 dim, since A's shape
                    # is [batch, spec=1, ...]
                    lA[k] = torch.cat([lA[k], v], dim=0)

        # merge all things from the batch
        lb = {k: torch.cat([item_lb[k] for item_lb in batch_lbs])
              for k in batch_lbs[0]}
        ub = {k: torch.cat([item_ub[k] for item_ub in batch_ubs])
              for k in batch_ubs[0]}

        return {
            'global_ub': ub[self.final_name], 'global_lb': lb[self.final_name],
            'mask': mask, 'lA': lA, 'lower_bounds': lb, 'upper_bounds': ub,
            'alphas': alphas, 'history': history, 'betas': ret_b,
            'input_split_idx': input_split_idx,
        }

    def build_history_and_set_bounds(self, d, split, mode='depth', impl_params=None):
        _, num_split = DomainUpdater.get_num_domain_and_split(
            d, split, self.final_name)
        input_split = getattr(self, 'new_input_split_now', False)

        args = (self.root, self.final_name, self.split_nodes)

        if impl_params is not None:
            domain_updater = DomainUpdaterImpliedCuts(*args, impl_params)
        elif input_split:
            domain_updater = DomainUpdaterInputSplit(*args)
        elif num_split == 1 and (split.get('points', None) is None
                                 or split['points'].ndim == 1):
            domain_updater = DomainUpdaterSimple(*args)
        else:
            domain_updater = DomainUpdater(*args)

        domain_updater.set_branched_bounds(d, split, mode)

    def _set_A_options(self, bab=False):
        branching_args = arguments.Config['bab']['branching']
        input_and_act = branching_args['branching_input_and_activation']
        fast_A = (branching_args['nonlinear_split']['method'] == 'shortcut'
                and branching_args['method'] == 'nonlinear')
        get_upper_bound = bab and arguments.Config['bab']['get_upper_bound']
        if get_upper_bound or input_and_act:
            self.needed_A_dict = defaultdict(set)
            self.needed_A_dict[self.net.output_name[0]].add(
                self.net.input_name[0])
        if fast_A:
            self.needed_A_dict = defaultdict(set)
            for node in self.net.nodes():
                if node != self.net.final_name:
                    self.needed_A_dict[node.name].add(self.net.input_name[0])
        # FIXME just use "self.needed_A_dict is not None" without the extra "self.return_A"
        if self.needed_A_dict is not None:
            self.return_A = True

    def empty_history(self):
        return {layer.name: ([], [], []) for layer in self.split_nodes}

    def _transfer(self, tensor, device=None, half=False):
        if half:
            tensor = tensor.half()
        if device:
            tensor = tensor.to(device)
        return tensor

    def set_crown_bound_opts(self, crown_name):
        solver_args = arguments.Config['solver']
        bab_args = arguments.Config['bab']
        crown_args = solver_args[f'{crown_name}-crown']
        opt_bound_args = {
            'deterministic': arguments.Config['general']['deterministic_opt'],
            'lr_alpha': crown_args['lr_alpha'],
            'iteration': crown_args['iteration'],
            'lr_decay': crown_args['lr_decay'],
            'use_float64_in_last_iteration': solver_args['use_float64_in_last_iteration'],
            'start_save_best': solver_args['start_save_best'],
            'loss_reduction_func': reduction_str2func(
                arguments.Config['general']['loss_reduction_func']),
        }
        if crown_name == 'alpha':
            opt_bound_args.update({
                'enable_alpha_crown': True, 'enable_beta_crown': False,
                'init_alpha': False, 'fix_interm_bounds': True,
                'use_shared_alpha': crown_args['share_alphas'],
                'early_stop_patience': solver_args['early_stop_patience'],
                'pruning_in_iteration': False
            })
        elif crown_name == 'beta':
            opt_bound_args.update({
                'optimizer': crown_args['optimizer'],
                'lr_beta': crown_args['lr_beta'],
                'pruning_in_iteration': bab_args['pruning_in_iteration'],
                'pruning_in_iteration_threshold': bab_args['pruning_in_iteration_ratio'],
                'lr_cut_beta': bab_args['cut']['lr_beta'],
                'include_output_constraint': False,
            })
        self.net.set_bound_opts({'optimize_bound_args': opt_bound_args})

    def _print_interm_bounds(self, lb=None, ub=None):
        if lb is not None:
            for k in lb:
                print(self.net[k])
                for out in self.net[k].output_name:
                    print('  ', self.net[out])
                print('  lower:', lb[k])
                if ub is not None and k in ub:
                    print('  upper:', ub[k])
                unstable = torch.logical_and(lb[k] < 0, ub[k] > 0).float()
                unstable = unstable.reshape(unstable.size(0), -1).sum(dim=-1)
                print('  unstable:', unstable, unstable.mean(), unstable.max())
        else:
            for node in self.net.nodes():
                if not node.perturbed:
                    continue
                if isinstance(getattr(node, 'lower', None), torch.Tensor):
                    print(node)
                    print('  lower:', node.lower.reshape(-1))
                    if isinstance(getattr(node, 'upper', None), torch.Tensor):
                        print('  upper:', node.upper.reshape(-1))
                        print(' Average gap:', (node.upper-node.lower).mean())

    from alpha import get_alpha, set_alpha, copy_alpha
    from beta import get_beta, set_beta, reset_beta
    from lp_mip_solver import (
        build_solver_model, update_mip_model_fix_relu,
        build_the_model_mip_refine, build_the_model_lp, build_the_model_mip,
        all_node_split_LP, check_lp_cut, update_the_model_cut)
    from input_split.batch_branch_and_bound import get_lower_bound_naive
    from cuts.cut_verification import (
        enable_cuts, create_cutter, set_cuts, create_mip_building_proc,
        set_cut_params, set_cut_new_split_history,
        disable_cut_for_branching, set_dependencies)
    from prune import prune_reference_alphas, prune_lA
