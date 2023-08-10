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
"""α,β-CROWN (alpha-beta-CROWN) verifier main interface."""

import copy
import socket
import random
import os
import time
import gc
import torch
import numpy as np
from collections import defaultdict

import arguments
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import stop_criterion_all, stop_criterion_batch_any
from jit_precompile import precompile_jit_kernels
from beta_CROWN_solver import LiRPANet
from lp_mip_solver import mip
from attack import attack
from utils import Logger, print_nonlinearities
from specifications import trim_batch, batch_vnnlib, sort_targets, prune_by_idx
from loading import load_model_and_vnnlib, parse_run_mode, adhoc_tuning, Customized  # pylint: disable=unused-import
from bab import general_bab
from input_split.batch_branch_and_bound import input_bab_parallel
from read_vnnlib import read_vnnlib
from cuts.cut_utils import terminate_mip_processes, terminate_mip_processes_by_c_matching
from lp_test import compare_optimized_bounds_against_lp_bounds


def incomplete_verifier(model_ori, data, data_ub=None, data_lb=None, vnnlib=None):
    # Generally, c should be constructed from vnnlib
    assert len(vnnlib) == 1, 'incomplete_verifier only support single x spec'
    input_x, specs = vnnlib[0]
    c_transposed = False
    if len(specs) > 1:
        # single OR with many clauses (e.g., robustness verification)
        assert all([len(_[0]) == 1 for _ in specs]), \
            'for each property in OR, only one clause supported so far'
        c = torch.concat([
            item[0] if isinstance(item[0], torch.Tensor) else torch.tensor(item[0])
            for item in specs], dim=0).unsqueeze(1).to(data)  # c shape: (batch, 1, num_outputs)
        if c.shape[0] != 1 and data.shape[0] == 1:
            # transpose c to shape (1,batch,num_outputs) to share intermediate bounds
            c = c.transpose(0, 1)
            c_transposed = True
        rhs = torch.tensor(np.array([item[1] for item in specs])).to(data).t()  # (batch, 1)
        stop_func = stop_criterion_all(rhs)

    else:
        # single AND with many clauses (e.g., Yolo).
        # shape: (batch=1, num_clauses in AND, num_outputs)
        c = torch.tensor(specs[0][0]).unsqueeze(0).to(data)
        # shape: (num_clauses in AND, 1)
        rhs = torch.tensor(specs[0][1], dtype=data.dtype, device=data.device).unsqueeze(0)
        stop_func = stop_criterion_batch_any(rhs)

    model = LiRPANet(model_ori, in_size=data.shape, c=c)
    if arguments.Config['solver']['alpha-crown']['include_output_constraint']:
        model.net.constraints = torch.tensor([x[0] for x in specs])
        assert model.net.constraints.ndim == 3
        assert model.net.constraints.size(0) == 1
        model.net.constraints = model.net.constraints.squeeze(0)
        model.net.thresholds = rhs

        # The output constraint implementation is currently specific for the yolo benchmark
        assert len(specs) == 1

        # We need to use matrix mode for the layer that should utilize output constraints
        assert model.net['/92'].mode == 'patches'
        model.net['/92'].mode = 'matrix'

    if isinstance(input_x, dict):
        # Traditional Lp norm case. Still passed in as an vnnlib variable, but it is passed
        # in as a dictionary.
        ptb = PerturbationLpNorm(
            norm=input_x['norm'],
            eps=input_x['eps'], eps_min=input_x.get('eps_min', 0),
            x_L=data_lb, x_U=data_ub)
    else:
        norm = arguments.Config['specification']['norm']
        # Perturbation value for non-Linf perturbations, None for all other cases.
        ptb = PerturbationLpNorm(norm=norm, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data.device)
    output = model.net(x)
    print('Model:', model.net)
    print('Original output:', output)
    if os.environ.get('ABCROWN_VIEW_MODEL', 0):
        print_nonlinearities(model.net)
        import pdb; pdb.set_trace()
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)
    bound_prop_method = arguments.Config['solver']['bound_prop_method']
    # one of them is sufficient.
    global_lb, ret = model.build(
        domain, x, stop_criterion_func=stop_func, decision_thresh=rhs, vnnlib_ori=vnnlib)

    if c_transposed:
        # transpose back to get ready for general verified condition check and final outputs
        global_lb = global_lb.t()
        rhs = rhs.t()

    if torch.any((global_lb - rhs) > 0, dim=-1).all():
        # Any spec in AND verified means verified. Also check all() at batch dim.
        print('verified with init bound!')
        return 'safe-incomplete', {}

    if arguments.Config['attack']['pgd_order'] == 'middle':
        if ret['attack_images'] is not None:
            return 'unsafe-pgd', {}

    # Save the alpha variables during optimization. Here the batch size is 1.
    saved_alphas = defaultdict(dict)
    for m in model.net.optimizable_activations:
        for spec_name, alpha in m.alpha.items():
            # each alpha size is (2, spec, 1, *shape); batch size is 1.
            saved_alphas[m.name][spec_name] = alpha.detach().clone()

    # FIXME there may be some duplicate with saved_alphas
    if bound_prop_method == 'alpha-crown':
        ret['activation_opt_params'] = {
            node.name: node.dump_optimized_params()
            for node in model.net.optimizable_activations
        }

    if c_transposed:
        ret['lower_bounds'][model.final_name] = ret['lower_bounds'][model.final_name].t()
        ret['upper_bounds'][model.final_name] = ret['upper_bounds'][model.final_name].t()
        if ret['lA'] is not None:
            ret['lA'] = {k: v.transpose(0, 1) for k, v in ret['lA'].items()}

    ret.update({'model': model, 'global_lb': global_lb, 'alpha': saved_alphas})

    return 'unknown', ret


def bab(unwrapped_model, data, targets, time_stamp, data_ub, data_lb, data_dict,
        lower_bounds=None, upper_bounds=None, reference_alphas=None,
        attack_images=None, c=None, cplex_processes=None,
        activation_opt_params=None, reference_lA=None, rhs=None,
        model_incomplete=None, timeout=None, refined_betas=None, vnnlib=None):
    # This will use the refined bounds if the complete verifier is "bab-refine".
    # FIXME do not repeatedly create LiRPANet which creates a new BoundedModule each time.

    rhs_offset = arguments.Config['debug']['rhs_offset']
    if rhs_offset is not None:
        print('Add an offset to RHS for debugging:', rhs_offset)
        rhs = rhs + rhs_offset

    # if using input split, transpose C if there are multiple specs with shared input,
    # to improve efficiency when calling the incomplete verifier later
    if arguments.Config['bab']['branching']['input_split']['enable']:
        c_transposed = False
        if data_lb.shape[0] == 1 and data_ub.shape[0] == 1 and \
                c is not None and c.shape[0] > 1 and c.shape[1] == 1:
            # multiple c instances (multiple vnnlibs) since c.shape[0] > 1,
            # but they share the same input (since data.shape[0] == 1）and
            # only single spec in each instance (c.shape[1] == 1)
            c = c.transpose(0, 1)
            rhs = rhs.transpose(0, 1)
            c_transposed = True

    model = LiRPANet(
        unwrapped_model, c=c, cplex_processes=cplex_processes,
        in_size=(data.shape if not len(targets) > 1
                 else [len(targets)] + list(data.shape[1:])))
    data = data.to(model.device)
    data_lb, data_ub = data_lb.to(model.device), data_ub.to(model.device)

    norm = arguments.Config['specification']['norm']
    if data_dict is not None:
        assert isinstance(data_dict['eps'], float)
        ptb = PerturbationLpNorm(
            norm=norm, eps=data_dict['eps'],
            eps_min=data_dict.get('eps_min', 0), x_L=data_lb, x_U=data_ub)
    else:
        ptb = PerturbationLpNorm(norm=norm, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)

    output = model.net(x).flatten()
    print('Model prediction is:', output)
    if not model_incomplete:
        print('Model:', model.net)

    if arguments.Config['attack']['check_clean']:
        clean_rhs = c.matmul(output)
        print(f'Clean RHS: {clean_rhs}')
        if (clean_rhs < rhs).any():
            return -torch.inf, None, 'unsafe'

    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)
    if arguments.Config['bab']['branching']['input_split']['enable']:
        min_lb, nb_states, verified_ret = input_bab_parallel(
            model, domain, x, model_ori=unwrapped_model, rhs=rhs, timeout=timeout,
            vnnlib=vnnlib, c_transposed=c_transposed)
    else:
        min_lb, nb_states, verified_ret = general_bab(
            model, domain, x,
            refined_lower_bounds=lower_bounds, refined_upper_bounds=upper_bounds,
            activation_opt_params=activation_opt_params, reference_lA=reference_lA,
            reference_alphas=reference_alphas, attack_images=attack_images,
            timeout=timeout, refined_betas=refined_betas, rhs=rhs,
            model_incomplete=model_incomplete, time_stamp=time_stamp)

    if min_lb is None:
        min_lb = -torch.inf
    elif isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()

    return min_lb, nb_states, verified_ret


def complete_verifier(
        model_ori, model_incomplete, vnnlib, batched_vnnlib, vnnlib_shape,
        index, timeout_threshold, bab_ret=None, cplex_processes=None,
        attack_images=None, attack_margins=None, results=None):

    start_time = time.time()

    enable_incomplete = arguments.Config['general']['enable_incomplete_verification']
    init_global_lb = results.get('global_lb', None)
    lower_bounds = results.get('lower_bounds', None)
    upper_bounds = results.get('upper_bounds', None)
    reference_alphas = results.get('alpha', None)
    lA = results.get('lA', None)
    cplex_cuts = (arguments.Config['bab']['cut']['enabled']
                  and arguments.Config['bab']['cut']['cplex_cuts'])
    bab_attack_enabled = arguments.Config['bab']['attack']['enabled']

    if enable_incomplete:
        final_name = model_incomplete.final_name
        init_global_ub = upper_bounds[final_name]
        print('lA shape:', [lAitem.shape for lAitem in lA.values()])
        (batched_vnnlib, init_global_lb, init_global_ub,
         lA, attack_images) = sort_targets(
            batched_vnnlib, init_global_lb, init_global_ub,
            attack_images, attack_margins, results, model_incomplete)
        if reference_alphas is not None:
            reference_alphas_cp = copy.deepcopy(reference_alphas)

    solved_c_rows = []

    time_stamp = 0
    for property_idx, properties in enumerate(batched_vnnlib):  # loop of x
        # batched_vnnlib: [x, [(c, rhs, y, pidx)]]
        print(f'\nProperties batch {property_idx}, size {len(properties[0])}')
        timeout = timeout_threshold - (time.time() - start_time)
        print(f'Remaining timeout: {timeout}')
        start_time_bab = time.time()

        if isinstance(properties[0][0], dict):
            def _get_item(properties, key):
                return torch.concat([
                    item[key].unsqueeze(0) for item in properties[0]], dim=0)
            x = _get_item(properties, 'X')
            data_min = _get_item(properties, 'data_min')
            data_max = _get_item(properties, 'data_max')
            # A dict to store extra variables related to the data and specifications
            for item in properties[0]:
                assert item['eps'] == properties[0][0]['eps']
            data_dict = {
                'eps': properties[0][0]['eps'],
                'eps_min': properties[0][0].get('eps_min', 0),
            }
        else:
            x_range = torch.tensor(properties[0], dtype=torch.get_default_dtype())
            data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
            data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
            x = x_range.mean(-1).reshape(vnnlib_shape)  # only the shape of x is important.
            data_dict = None

        target_label_arrays = list(properties[1])  # properties[1]: (c, rhs, y, pidx)
        assert len(target_label_arrays) == 1
        c, rhs, pidx = target_label_arrays[0]

        if bab_attack_enabled:
            if arguments.Config['bab']['initial_max_domains'] != 1:
                raise ValueError(
                    'To run Bab-attack, please set initial_max_domains to 1. '
                    f'Currently it is {arguments.Config["bab"]["initial_max_domains"]}.')
            # Attack images has shape (batch, restarts, specs, c, h, w).
            # The specs dimension should already be sorted.
            # Reshape it to (restarts, c, h, w) for this specification.
            this_spec_attack_images = attack_images[:, :, property_idx].view(
                attack_images.size(1), *attack_images.shape[3:])
        else:
            this_spec_attack_images = None

        # FIXME Clean up.
        # Shape and type of rhs is very confusing
        rhs = torch.tensor(rhs, device=arguments.Config['general']['device'],
                           dtype=torch.get_default_dtype())
        if (arguments.Config['general']['enable_incomplete_verification']
                and len(init_global_lb) > 1):
            # no need to trim_batch if batch = 1
            ret_trim = trim_batch(
                model_incomplete, init_global_lb, init_global_ub,
                reference_alphas_cp, lower_bounds, upper_bounds,
                reference_alphas, lA, property_idx, c, rhs)
            lA_trim, rhs = ret_trim['lA'], ret_trim['rhs']
        else:
            lA_trim = lA.copy() if lA is not None else lA

        print(f'##### Instance {index} first 10 spec matrices: ')
        print(f'{c[:10]}\nthresholds: {rhs.flatten()[:10]} ######')

        torch.cuda.empty_cache()
        gc.collect()
        c = c.to(rhs)  # both device and dtype

        # compress the first dim of data_min, data_max based on duplication check
        if data_min.shape[0] > 1:
            l1_err_data_min = torch.norm((data_min[1:] - data_min[0:1]).view(-1), p=1)
            l1_err_data_max = torch.norm((data_max[1:] - data_max[0:1]).view(-1), p=1)
            if l1_err_data_min + l1_err_data_max < 1e-8:
                # almost same x so we can use the first x
                x, data_min, data_max = x[0:1], data_min[0:1], data_max[0:1]

        # Complete verification (BaB, BaB with refine, or MIP).
        time_stamp += 1
        input_split = arguments.Config['bab']['branching']['input_split']['enable']
        init_failure_idx = np.array([])
        if enable_incomplete and not input_split:
            if len(init_global_lb) > 1:  # if batch == 1, there is no need to filter here.
                # Reuse results from incomplete results, or from refined MIPs.
                # skip the prop that already verified
                rlb = lower_bounds[final_name]
                # The following flatten is dangerous, each clause in OR only has one output bound.
                assert len(rlb.shape) == len(rhs.shape) == 2 and rlb.shape[1] == rhs.shape[1] == 1
                init_verified_cond = rlb.flatten() > rhs.flatten()
                init_verified_idx = torch.where(init_verified_cond)[0]
                if len(init_verified_idx) > 0:
                    print('Initial alpha-CROWN verified for spec index '
                            f'{init_verified_idx} with bound '
                            f'{rlb[init_verified_idx].squeeze()}.')
                    l = init_global_lb[init_verified_idx].tolist()
                    bab_ret.append([index, l, 0, time.time() - start_time_bab, pidx])
                init_failure_idx = torch.where(~init_verified_cond)[0]
                if len(init_failure_idx) == 0:
                    # This batch of x verified by init opt crown
                    continue
                print(f'Remaining spec index {init_failure_idx} with '
                        f'bounds {rlb[init_failure_idx]} need to verify.')

                reference_alphas, lA_trim, x, data_min, data_max, lower_bounds, upper_bounds, c \
                    = prune_by_idx(reference_alphas, init_verified_cond, final_name, lA_trim, x,
                                   data_min, data_max, lA is not None,
                                   lower_bounds, upper_bounds, c)

            l, nodes, ret = bab(
                model_ori, x, init_failure_idx, time_stamp=time_stamp,
                data_ub=data_max, data_lb=data_min, data_dict=data_dict,
                lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                c=c, reference_alphas=reference_alphas, cplex_processes=cplex_processes,
                activation_opt_params=results.get('activation_opt_params', None),
                refined_betas=results.get('refined_betas', None), rhs=rhs[0:1],
                reference_lA=lA_trim, attack_images=this_spec_attack_images,
                model_incomplete=model_incomplete, timeout=timeout, vnnlib=vnnlib)
            bab_ret.append([index, float(l), nodes,
                            time.time() - start_time_bab,
                            init_failure_idx.tolist()])
        else:
            assert arguments.Config['general']['complete_verifier'] == 'bab'
            assert not arguments.Config['bab']['attack']['enabled'], (
                'BaB-attack must be used with incomplete verifier.')
            # input split also goes here directly
            l, nodes, ret = bab(
                model_ori, x, pidx, time_stamp,
                data_ub=data_max, data_lb=data_min, c=c, data_dict=data_dict,
                cplex_processes=cplex_processes,
                rhs=rhs, timeout=timeout, attack_images=this_spec_attack_images, vnnlib=vnnlib)
            bab_ret.append([index, l, nodes, time.time() - start_time_bab, pidx])

        # terminate the corresponding cut inquiry process if exists
        if cplex_cuts:
            solved_c_rows.append(c)
            terminate_mip_processes_by_c_matching(cplex_processes, solved_c_rows)

        timeout = timeout_threshold - (time.time() - start_time)
        if ret == 'unsafe':
            return 'unsafe-bab'
        elif ret == 'unknown' or timeout < 0:
            return 'unknown'
        elif ret != 'safe':
            raise ValueError(f'Unknown return value of bab: {ret}')
    else:
        return 'safe'


def main():
    print(f'Experiments at {time.ctime()} on {socket.gethostname()}')
    torch.manual_seed(arguments.Config['general']['seed'])
    random.seed(arguments.Config['general']['seed'])
    np.random.seed(arguments.Config['general']['seed'])
    torch.set_printoptions(precision=8)
    device = arguments.Config['general']['device']
    if device != 'cpu':
        torch.cuda.manual_seed_all(arguments.Config['general']['seed'])
        # Always disable TF32 (precision is too low for verification).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if arguments.Config['general']['deterministic']:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
    if arguments.Config['general']['double_fp']:
        torch.set_default_dtype(torch.float64)
    if arguments.Config['general']['precompile_jit']:
        precompile_jit_kernels()

    bab_args = arguments.Config['bab']
    timeout_threshold = bab_args['timeout']
    select_instance = arguments.Config['data']['select_instance']
    (run_mode, save_path, file_root, example_idx_list, model_ori,
     vnnlib_all, shape) = parse_run_mode()
    logger = Logger(run_mode, save_path, timeout_threshold)

    for new_idx, csv_item in enumerate(example_idx_list):
        arguments.Globals['example_idx'] = new_idx
        vnnlib_id = new_idx + arguments.Config['data']['start']
        # Select some instances to verify
        if select_instance and not vnnlib_id in select_instance:
            continue
        logger.record_start_time()

        print(f'\n {"%"*35} idx: {new_idx}, vnnlib ID: {vnnlib_id} {"%"*35}')

        if run_mode != 'customized_data':
            # FIXME Don't write arguments.Config['bab']['timeout']!
            if len(csv_item) == 3:
                # model, vnnlib, timeout
                model_ori, shape, vnnlib, onnx_path = load_model_and_vnnlib(
                    file_root, csv_item)
                arguments.Config['model']['onnx_path'] = os.path.join(file_root, csv_item[0])
                arguments.Config['specification']['vnnlib_path'] = os.path.join(
                    file_root, csv_item[1])
            else:
                # Each line contains only 1 item, which is the vnnlib spec.
                vnnlib = read_vnnlib(os.path.join(file_root, csv_item[0]))
                assert arguments.Config['model']['input_shape'] is not None, (
                    'vnnlib does not have shape information, '
                    'please specify by --input_shape')
                shape = arguments.Config['model']['input_shape']
        else:
            vnnlib = vnnlib_all[new_idx]  # vnnlib_all is a list of all standard vnnlib

        # FIXME Don't write bab_args['timeout'] above.
        # Then these updates can be moved to arguments.update_arguments()
        bab_args['timeout'] = float(bab_args['timeout'])
        if bab_args['timeout_scale'] != 1:
            new_timeout = bab_args['timeout'] * bab_args['timeout_scale']
            print(f'Scaling timeout: {bab_args["timeout"]} -> {new_timeout}')
            bab_args['timeout'] = new_timeout
        if bab_args['override_timeout'] is not None:
            new_timeout = bab_args['override_timeout']
            print(f'Overriding timeout: {new_timeout}')
            bab_args['timeout'] = new_timeout
        timeout_threshold = bab_args['timeout']
        logger.update_timeout(timeout_threshold)

        if arguments.Config['general']['complete_verifier'].startswith('Customized'):
            res = eval(  # pylint: disable=eval-used
                        arguments.Config['general']['complete_verifier']
                        )(model_ori, vnnlib, os.path.join(file_root, onnx_path))
            logger.summarize_results(res, new_idx)
            continue

        model_ori.eval()
        vnnlib_shape = shape

        # FIXME attack and initial_incomplete_verification only works for assert len(vnnlib) == 1
        if isinstance(vnnlib[0][0], dict):
            x = vnnlib[0][0]['X'].reshape(vnnlib_shape)
            data_min = vnnlib[0][0]['data_min'].reshape(vnnlib_shape)
            data_max = vnnlib[0][0]['data_max'].reshape(vnnlib_shape)
        else:
            x_range = torch.tensor(vnnlib[0][0])
            data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
            data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
            x = x_range.mean(-1).reshape(vnnlib_shape)  # only the shape of x is important.
        adhoc_tuning(data_min, data_max, model_ori)

        model_ori = model_ori.to(device)
        x, data_max, data_min = x.to(device), data_max.to(device), data_min.to(device)
        verified_status, verified_success = 'unknown', False

        if arguments.Config['attack']['pgd_order'] == 'before':
            verified_status, verified_success, _, attack_margins, all_adv_candidates = attack(
                model_ori, x, vnnlib, verified_status, verified_success)
        else:
            attack_margins = all_adv_candidates = None

        model_incomplete = cplex_processes = None
        ret = {}

        if arguments.Config['debug']['test_optimized_bounds']:
            compare_optimized_bounds_against_lp_bounds(
                model_ori, x, data_ub=data_max, data_lb=data_min, vnnlib=vnnlib
            )

        # Incomplete verification is enabled by default. The intermediate lower
        # and upper bounds will be reused in bab and mip.
        if (not verified_success and (
                arguments.Config['general']['enable_incomplete_verification']
                or arguments.Config['general']['complete_verifier'] == 'bab-refine')):
            verified_status, ret = incomplete_verifier(
                model_ori, x, data_ub=data_max, data_lb=data_min, vnnlib=vnnlib)
            verified_success = verified_status != 'unknown'
            model_incomplete = ret.get('model', None)

        if not verified_success and arguments.Config['attack']['pgd_order'] == 'after':
            verified_status, verified_success, _, attack_margins, all_adv_candidates = attack(
                model_ori, x, vnnlib, verified_status, verified_success)

        # MIP or MIP refined bounds.
        if not verified_success and (
                arguments.Config['general']['complete_verifier'] == 'mip'
                or arguments.Config['general']['complete_verifier'] == 'bab-refine'):
            # rhs = ? NEED TO SAVE TO LIRPA_MODULE
            mip_skip_unsafe = arguments.Config['solver']['mip']['skip_unsafe']
            verified_status, ret_mip = mip(model_incomplete, ret, mip_skip_unsafe=mip_skip_unsafe)
            verified_success = verified_status != 'unknown'
            ret.update(ret_mip)

        # extract the process pool for cut inquiry
        if bab_args['cut']['enabled'] and bab_args['cut']['cplex_cuts']:
            assert arguments.Config['bab']['initial_max_domains'] == 1
            # use nullity of model_incomplete as an indicator of whether cut processes are launched
            if model_incomplete is not None:
                cplex_processes = model_incomplete.processes
                mip_building_proc = model_incomplete.mip_building_proc

        # BaB bounds. (not do bab if unknown by mip solver for now)
        if (not verified_success
                and arguments.Config['general']['complete_verifier'] != 'skip'
                and verified_status != 'unknown-mip'):
            batched_vnnlib = batch_vnnlib(vnnlib)  # [x, [(c, rhs, y, pidx)]] in batch-wise
            verified_status = complete_verifier(
                model_ori, model_incomplete, vnnlib, batched_vnnlib, vnnlib_shape,
                new_idx, bab_ret=logger.bab_ret, cplex_processes=cplex_processes,
                timeout_threshold=timeout_threshold - (time.time() - logger.start_time),
                attack_images=all_adv_candidates,
                attack_margins=attack_margins, results=ret)

        if (bab_args['cut']['enabled'] and bab_args['cut']['cplex_cuts']
                and model_incomplete is not None):
            terminate_mip_processes(mip_building_proc, cplex_processes)
            del cplex_processes
        del ret

        # Summarize results.
        logger.summarize_results(verified_status, new_idx)

    logger.finish()


if __name__ == '__main__':
    arguments.Config.parse_config()
    main()
