import os
import re
import torch
import numpy as np

inf = np.inf

def parse_tensor(lines):
    while not 'tensor(' in lines[0]:
        lines = lines[1:]
    full_line = ''
    for j in range(len(lines)):
        full_line += lines[j].strip()
        if ')' in lines[j]:
            break
    start = full_line.find('[')
    end = full_line.rfind(']')
    tensor = torch.tensor(eval(full_line[start:end+1])).view(-1)
    return tensor

def parse_initial_alpha_crown_bounds(lines, out):
    if lines[0].lower().startswith('initial alpha-crown bounds:'):
        out['init_alpha_crown'] = parse_tensor(lines)
        return True
    return False

def parse_refined_global_lb(lines, out):
    if lines[0].startswith('refined global lb:'):
        out['refined_lb'] = parse_tensor(lines)
        return True
    return False

def parse_model_prediction(lines, out):
    if lines[0].startswith('Model prediction is') or lines[0].startswith('Original output'):
        if '(first 10)' in lines[0]:
            lines[0] = lines[0].replace('(first 10)', ' ')
        out['pred'] = parse_tensor(lines)
        return True
    return False

def parse_splitting_decisions(lines, out):
    # Not able to compare for now due to newly added 'split level'
    return False

def parse_common(lines, out):
    if parse_model_prediction(lines, out):
        return True
    if parse_initial_alpha_crown_bounds(lines, out):
        return True
    if parse_refined_global_lb(lines, out):
        return True
    if parse_splitting_decisions(lines, out):
        return True

    line = lines[0]

    if 'Time out!!!!!!!!' in line:
        out['results'] = 'timeout'
        return True
    if 'Error' in line and 'ONNXRuntimeError' not in line and 'Error traceback' not in line:
        # make sure this is not a onnx conversion correctness check error
        out['results'] = 'error'
    if out['results'] == 'timeout':
        if 'adversarial example found' in line:
            out['results'] = 'unsafe'
        if ('total verified: 1' in line or 'verified safe: 1' in line
                or 'total verified (safe/unsat): 1' in line):
            out['results'] = 'safe'
        if ('verified unsafe: 1' in line
                or 'total falsified (unsafe/sat): 1' in line):
            out['results'] = 'unsafe'

    if ((line.startswith('mean time') or line.startswith('max time'))
            and line[-1].isdigit()):
        if 'Warning' in line and '/' in line:
            out['time'] = float(line[:line.find('/')].split()[-1])
        elif 'Warning' in line and '<' in line:
            out['time'] = float(line[:line.find('<')].split()[-1])
        else:
            out['time'] = float(line.split()[-1])

    return False

def parse_log(file_name):
    out = {'idx': None, 'pred': None, 'init_alpha_crown': None,
            'refined_lb': None, 'decisions': [], 'results': 'timeout'}
    buffer_lines = 100
    with open(file_name) as f:
        all_lines = f.readlines()
        for i, line in enumerate(all_lines):
            if parse_common(all_lines[i:i+buffer_lines], out):
                continue
            if 'idx:' in line:
                m = re.match(r'.*idx: (\d+).*', line)
                out['idx'] = int(m.group(1))
                continue
            if line.endswith('neurons visited\n'):
                neurons_visited = int(line.split()[0])
                out['neurons_visited'] = max(out.get('neurons_visited', 0), neurons_visited)
    return out

def parse_compare(benchmark, source, reference, auto_test=False,
        ignore_time=False, ignore_decisions=False, ignore_visited_neurons=False):
    # auto_test: for automatic GPU test which may tolerate more difference

    passed = True

    source_folder = os.path.join(benchmark, source)
    reference_folder = os.path.join(benchmark, 'references', reference)
    if not os.path.exists(source_folder):
        print(f'Warning: source folder {source_folder} not exists, skip!')
        return
    if not os.path.exists(reference_folder):
        print(f'Warning: reference folder {reference_folder} not exists, skip!')
        return

    source_log_files = []
    reference_log_files = []
    for filename in os.listdir(reference_folder):
        if not filename.endswith(".out"):
            continue
        reference_log_files.append(filename)
        # For compatibility with legacy reference files
        if filename.startswith('test'):
            source_log_files.append(f"{int(''.join(filter(str.isdigit, filename)))-1}.out")
        else:
            source_log_files.append(filename)
    for s, r in zip(source_log_files, reference_log_files):
        error_keys = []
        error_msg = []
        s = os.path.join(source_folder, s)
        r = os.path.join(reference_folder, r)
        assert os.path.exists(s), f'{s} not exists'
        assert os.path.exists(r), f'{r} not exists'
        print(f' === source: {s} <- reference: {r} === ')
        sout = parse_log(s)
        rout = parse_log(r)
        checked = True
        for key in sout:
            svalue, rvalue = sout[key], rout[key]
            if isinstance(svalue, torch.Tensor):
                try:
                    if auto_test:
                        if key in ['init_alpha_crown']:
                            checked = torch.allclose(svalue, rvalue, rtol=1e-1, atol=0.05)
                        elif key in ['refined_lb']:
                            checked = True # Skip
                        else:
                            checked = torch.max(torch.abs(svalue - rvalue)) <= 1e-3
                    else:
                        checked = torch.max(torch.abs(svalue - rvalue)) <= 1e-3
                except:
                    checked = False
                if not checked:
                    error_keys.append(key)
                    error_msg.append(f'{key}: {svalue} <- {rvalue}')
            elif key == 'decisions':
                if ignore_decisions:
                    continue
                for sd, rd in zip(svalue, rvalue):
                    try:
                        checked = torch.max(torch.abs(sd - rd)) == 0
                    except:
                        checked = False
                    if not checked:
                        error_keys.append(key)
                        error_msg.append(f'{key}: {sd} <- {rd}')
            elif key == 'time':
                if not ignore_time and abs(svalue - rvalue) > 10 and (svalue - rvalue) / rvalue > 0.2:
                    checked = False
                    error_keys.append(key)
                    error_msg.append(f'{key}: {svalue} <- {rvalue}')
            elif key == 'neurons_visited':
                if not ignore_visited_neurons and abs(svalue - rvalue) / rvalue > 0.2:
                    checked = False
                    error_keys.append(key)
                    error_msg.append(f'{key}: {svalue} <- {rvalue}')
            else:
                try:
                    checked = svalue == rvalue
                    if isinstance(checked, torch.Tensor):
                        checked = checked.all()
                except:
                    checked = False
                if not checked:
                    error_keys.append(key)
                    error_msg.append(f'{key}: {svalue} <- {rvalue}')
        if error_keys:
            print(f'[check fail]')
            for msg in error_msg:
                print(msg)
            passed = False
        else:
            print(f'[check pass]')

    return passed
