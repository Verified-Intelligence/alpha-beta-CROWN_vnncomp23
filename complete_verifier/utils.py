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

import copy
import os
import time
import pickle
import arguments
from dataclasses import dataclass
from collections import defaultdict
from string import Template
import torch


@dataclass
class Timer:
    total_func_time: float = 0.0
    total_prepare_time: float = 0.0
    total_bound_time: float = 0.0
    total_beta_bound_time: float = 0.0
    total_transfer_time: float = 0.0
    total_finalize_time: float = 0.0

    def __init__(self):
        self.time_start = {}
        self.time_last = {}
        self.time_sum = {}

    def start(self, name):
        self.time_start[name] = time.time()
        if name not in self.time_sum:
            self.time_sum[name] = 0

    def add(self, name):
        self.time_last[name] = time.time() - self.time_start[name]
        self.time_sum[name] += self.time_last[name]

    def print(self):
        print('Time: ', end='')
        for k, v in self.time_last.items():
            print(f'{k} {v:.4f}', end='    ')
        print()
        print('Accumulated time: ', end='')
        for k, v in self.time_sum.items():
            print(f'{k} {v:.4f}', end='    ')
        print()


class Logger:
    def __init__(self, run_mode, save_path, timeout_threshold):
        self.run_mode = run_mode
        self.save_path = save_path
        self.timeout_threshold = timeout_threshold
        self.verification_summary = defaultdict(list)
        self.time_all_instances = []
        self.status_per_sample_list = []
        self.bab_ret = []
        self.count = 0

    def update_timeout(self, timeout):
        self.timeout_threshold = timeout

    def record_start_time(self):
        self.start_time = time.time()

    def summarize_results(self, verified_status, index):
        self.count += 1
        if self.run_mode == 'single_vnnlib':
            # run in run_instance.sh
            if ('unknown' in verified_status or 'timeout' in verified_status
                or 'timed out' in verified_status):
                verified_status = 'timeout'
            elif 'unsafe' in verified_status:
                verified_status = 'sat'
            elif 'safe' in verified_status:
                verified_status = 'unsat'
            else:
                raise ValueError(f'Unknown verified_status {verified_status}')

            print('Result:', verified_status)
            print('Time:', time.time() - self.start_time)
            with open(self.save_path, 'w') as file:
                file.write(verified_status)
                if arguments.Config['general']['save_adv_example']:
                    if verified_status == 'sat':
                        file.write('\n')
                        cex_path = arguments.Config['attack']['cex_path']
                        with open(cex_path, 'r') as adv_example:
                            file.write(adv_example.read())
                file.flush()
        else:
            if time.time() - self.start_time > self.timeout_threshold:
                if 'unknown' not in verified_status:
                    verified_status += ' (timed out)'
            self.verification_summary[verified_status].append(index)
            self.status_per_sample_list.append(
                [verified_status, time.time() - self.start_time])
            self._save()
            print(f'Result: {verified_status} '
                  f'in {self.status_per_sample_list[-1][1]:.4f} seconds')

    def finish(self):
        if self.run_mode != 'single_vnnlib':
            # Finished all examples.
            time_timeout = [
                s[1] for s in self.status_per_sample_list if 'unknown' in s[0]]
            time_verified = [
                s[1] for s in self.status_per_sample_list
                if 'safe' in s[0] and 'unsafe' not in s[0]]
            time_unsafe = [
                s[1] for s in self.status_per_sample_list if 'unsafe' in s[0]]
            time_all_instances = [s[1] for s in self.status_per_sample_list]
            self._save()

            print('############# Summary #############')
            acc = len(time_verified) / self.count * 100.
            print(f'Final verified acc: {acc}% (total {self.count} examples)')
            print('Problem instances count:',
                  len(time_verified) + len(time_unsafe) + len(time_timeout),
                  ', total verified (safe/unsat):', len(time_verified),
                  ', total falsified (unsafe/sat):', len(time_unsafe),
                  ', timeout:', len(time_timeout))
            print('mean time for ALL instances '
                  f'(total {len(time_all_instances)}):'
                  f'{sum(time_all_instances)/(len(time_all_instances) + 1e-5)},'
                  f' max time: {max(time_all_instances)}')
            if len(time_verified) > 0:
                print('mean time for verified SAFE instances'
                      f'(total {len(time_verified)}): '
                      f'{sum(time_verified) / len(time_verified)}, '
                      f'max time: {max(time_verified)}')
            if len(time_verified) > 0 and len(time_unsafe) > 0:
                mean_time = (sum(time_verified) + sum(time_unsafe)) / (
                    len(time_verified) + len(time_unsafe))
                max_time = max(time_verified, time_unsafe)
                print('mean time for verified (SAFE + UNSAFE) instances '
                      f'(total {(len(time_verified) + len(time_unsafe))}):'
                      f' {mean_time}, max time: {max_time}')
            if len(time_verified) > 0 and len(time_timeout) > 0:
                mean_time = (sum(time_verified) + sum(time_timeout)) / (
                    len(time_verified) + len(time_timeout))
                max_time = max(time_verified, time_timeout)
                print('mean time for verified SAFE + TIMEOUT instances '
                      f'(total {(len(time_verified) + len(time_timeout))}):'
                      f' {mean_time}, max time: {max_time} ')
            if len(time_unsafe) > 0:
                print(f'mean time for verified UNSAFE instances '
                      f'(total {len(time_unsafe)}): '
                      f'{sum(time_unsafe) / len(time_unsafe)}, '
                      f'max time: {max(time_unsafe)}')

            for k, v in self.verification_summary.items():
                print(f'{k} (total {len(v)}), index:', v)

    def _save(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump({
                'summary': self.verification_summary,
                'results': self.status_per_sample_list,
                'bab_ret': self.bab_ret
            }, f)


class Stats:
    def __init__(self):
        self.visited = 0
        self.timer = Timer()
        self.all_node_split = False
        self.implied_cuts = {'statistics': [], 'average_branched_neurons': []}


def get_reduce_op(op):
    """Convert reduce op in str to the actual function."""
    if op is None:
        return op
    elif op in ['min', 'max', 'mean']:
        return getattr(torch, op)
    else:
        raise ValueError(op)


def fast_hist_copy(hists):
    """Copy the history for one element. Much faster than deepcopy()."""
    if hists is None:
        return None
    ret = {}
    for k, hist in hists.items():
        if isinstance(hist[0], torch.Tensor):
            ret[k] = hist
        elif isinstance(hist[0], list):
            ret[k] = tuple(hist[i].copy() for i in range(3))
        else:
            ret[k] = tuple(copy.deepcopy(hist[i]) for i in range(3))
    return ret


def print_splitting_decisions(net, d, split_depth, split):
    """Print the first two split for first 10 domains."""
    print('splitting decisions: ')
    branching_decision = split['decision']
    for l in range(split_depth):
        print(f'split level {l}', end=': ')
        for b in range(min(10, len(d['history']))):
            decision = branching_decision[l*len(d['history']) + b]
            print(f'[{net.split_nodes[decision[0]].name}, {decision[1]}]',
                  end=' ')
        print('')


def print_average_branching_neurons(branching_decision, impl_stats, impl_params=None):
    """Print and store the average branched neurons at each iteration."""
    total_branched_neurons = 0

    if impl_params:
        components = impl_params['dependency_components']
        idx_mapping = impl_params['index_mappings']
        for neurons in branching_decision:
            core_idx = idx_mapping[(neurons[0], neurons[1])]
            total_branched_neurons += len(components[core_idx][2])
        average_branched_neurons = total_branched_neurons / len(branching_decision)
    else:
        average_branched_neurons = 1.0

    impl_stats['average_branched_neurons'].append(average_branched_neurons)
    cur_step = len(impl_stats['average_branched_neurons'])

    print(f'Average branched neurons at iteration {cur_step}: '
          f'{average_branched_neurons: .4f}')


def repeat(x, num_copy, unsqueeze=False):
    """Repeat a tensor by the first dimension."""
    if x is None:
        return None
    if isinstance(x, list):
        return x * num_copy
    if unsqueeze:
        return x.unsqueeze(0).repeat(num_copy, *[1]*x.ndim)
    else:
        return x.repeat(num_copy, *[1]*(x.ndim - 1))


def check_infeasible_bounds(lower, upper, reduce=False):
    print('Checking infeasibility')
    infeasible = None
    for k in lower:
        infeasible_ = (lower[k] - upper[k]).view(
            lower[k].shape[0], -1).max(dim=-1).values > 1e-6
        # FIXME check_infeasible_bounds first before moving the bounds to CPU
        infeasible_ = infeasible_.cpu()
        infeasible = (infeasible_ if infeasible is None
                      else torch.logical_or(infeasible, infeasible_))
    any_infeasible = infeasible.any()
    if any_infeasible:
        print(f'Infeasiblity detected! {int(infeasible.sum())} domains')
    if reduce:
        return any_infeasible
    else:
        return infeasible


def get_save_path(csv):
    if csv:
        return 'a-b-crown_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}_cplex_cuts={}_initial_max_domains={}.npz'.format(  # pylint: disable=line-too-long,consider-using-f-string
            os.path.splitext(os.path.basename(arguments.Config['general']['csv_name']))[0],
            arguments.Config['data']['start'],
            arguments.Config['data']['end'], arguments.Config['solver']['beta-crown']['iteration'],
            arguments.Config['solver']['batch_size'],
            arguments.Config['bab']['timeout'], arguments.Config['bab']['branching']['method'],
            arguments.Config['bab']['branching']['reduceop'],
            arguments.Config['bab']['branching']['candidates'],
            arguments.Config['solver']['alpha-crown']['lr_alpha'],
            arguments.Config['solver']['beta-crown']['lr_alpha'],
            arguments.Config['solver']['beta-crown']['lr_beta'],
            arguments.Config['attack']['pgd_order'],
            arguments.Config['bab']['cut']['cplex_cuts'],
            arguments.Config['bab']['initial_max_domains'])
    else:
        if arguments.Config['model']['name'] is None:
            # use onnx model prefix as model_name
            model_name = arguments.Config['model']['onnx_path'].split('.onnx')[-2].split('/')[-1]
        elif 'Customized' in arguments.Config['model']['name']:
            model_name = 'Customized_model'
        else:
            model_name = arguments.Config['model']['name']
        return 'Verified_ret_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}_cplex_cuts={}.npy'.format(  # pylint: disable=line-too-long,consider-using-f-string
            model_name, arguments.Config['data']['start'], arguments.Config['data']['end'],
            arguments.Config['solver']['beta-crown']['iteration'],
            arguments.Config['solver']['batch_size'],
            arguments.Config['bab']['timeout'], arguments.Config['bab']['branching']['method'],
            arguments.Config['bab']['branching']['reduceop'],
            arguments.Config['bab']['branching']['candidates'],
            arguments.Config['solver']['alpha-crown']['lr_alpha'],
            arguments.Config['solver']['beta-crown']['lr_alpha'],
            arguments.Config['solver']['beta-crown']['lr_beta'],
            arguments.Config['attack']['pgd_order'], arguments.Config['bab']['cut']['cplex_cuts'])


def print_nonlinearities(model):
    print('Nonlinearities:')
    for node in model.nodes():
        if node.perturbed and node.requires_input_bounds:
            print(node)


def get_batch_size_from_masks(mask):
    return len(next(iter(mask.values())))


def get_unstable_neurons(updated_mask):
    tot_ambi_nodes = 0
    # only pick the first copy from possible multiple x
    updated_mask = {k: [item[0:1] if item is not None else None
                        for item in mask]
                    for k, mask in updated_mask.items()}
    for k, masks in updated_mask.items():
        for i, mask in enumerate(masks):
            if mask is None: # Not perturbed
                continue
            n_unstable = int(torch.sum(mask).item())
            print(f'Node {k} input {i}: size {mask.shape[1:]} unstable {n_unstable}')
            tot_ambi_nodes += n_unstable
    print(f'-----------------\n# of unstable neurons: {tot_ambi_nodes}\n-----------------\n')
    return updated_mask, tot_ambi_nodes


def expand_path(path):
    return Template(path).substitute(
        CONFIG_PATH=os.path.dirname(arguments.Config.file))
