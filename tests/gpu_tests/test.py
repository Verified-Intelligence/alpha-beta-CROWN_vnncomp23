import os
import argparse
import shutil
import yaml
from log_parser import parse_compare

parser = argparse.ArgumentParser()
parser.add_argument('--run', '-r', action='store_true')
parser.add_argument('--testset', '-s', type=str, nargs='+', default=[])
parser.add_argument('--benchmark', '-b', type=str, nargs='+', default=[], help='Only run some benchmarks')
parser.add_argument('--reference', type=str, help='A reference to be specified for checking')
parser.add_argument('--install-ref', type=str, help='Install a reference from another copy of the code. Specify the name of the reference.')
parser.add_argument('--ignore-time', action='store_true', help='Ignore comparison on time')
parser.add_argument('--ignore-decisions', action='store_true', help='Ignore comparison on branching decisions')
parser.add_argument('--ignore-visited-neurons', action='store_true', help='Ignore comparison on the number of visited neurons')
parser.add_argument('--idx', type=int, nargs='+', help='Only run a specific test case.')
parser.add_argument('--remove_old_output', action='store_true', help='Remove old master output files.')
args = parser.parse_args()

def get_all_tests(dir='.'):
    tests = []
    for f in os.listdir(dir):
        f_ = os.path.join(dir, f)
        if f != 'master_outputs' and os.path.isdir(f_):
            tests += get_all_tests(f_)
    if (os.path.exists(os.path.join(dir, 'run_master.sh'))
            or os.path.exists(os.path.join(dir, 'config.yaml'))):
        if dir.startswith('./'):
            dir = dir[2:]
        tests.append(dir)
    return tests

def run_test(dir):
    """ Iterate over all directories and run all `run_master.sh`"""
    print(f'Running test: {dir}')
    out_dir = os.path.join(dir, 'master_outputs')
    if args.remove_old_output and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(dir, 'tests.yaml')) as file:
        tests = yaml.safe_load(file.read())
    tests = tests['tests']

    for i, test in enumerate(tests):
        if args.idx is not None and i not in args.idx:
            continue
        print(f'Test case {i}')
        idx = test['idx']
        test_args = test.get('args', '')
        arguments = f'--start {idx} --end {idx+1} {test_args}'
        print(f'  arguments: {arguments}')
        command = 'PYTHONPATH=' + os.path.abspath('../../') + ' '
        command += test.get('env', '') + ' '
        command += ' python '
        command += os.path.abspath('../../complete_verifier/abcrown.py')
        command += ' --config ' + os.path.abspath(os.path.join(dir, 'config.yaml'))
        command += f' {arguments}'
        command += ' 2>&1 | tee ' + os.path.abspath(os.path.join(dir, 'master_outputs', f'{i}.out'))
        print(f'  {command}')
        os.system(f'cd {dir} && {command}')
        print()
    print()

def check_test(dir):
    print(f'Checking {dir}')
    if args.reference:
        ref = args.reference
    else:
        ref_latest = None
        for ref in os.listdir(os.path.join(dir, 'references')):
            if ref.startswith('202'):
                if ref_latest is None or ref > ref_latest:
                    ref_latest = ref
        assert ref_latest is not None, f'Reference not found in {dir}'
        ref = ref_latest
    print(f'Reference: {ref}')
    res = parse_compare(dir, 'master_outputs', ref, auto_test=True,
        ignore_time=args.ignore_time, ignore_decisions=args.ignore_decisions, ignore_visited_neurons=args.ignore_visited_neurons)
    print('Check result:', res)
    print()
    return res

def install_ref(tests, ref_name):
    for item in tests:
        ref_dir = f'{item}/references/{ref_name}'
        if not os.path.exists(ref_dir):
            os.makedirs(ref_dir)
        command = f'cp -r ./{item}/master_outputs/* {ref_dir}'
        os.system(command)
        print(f'Reference copied to {ref_dir}')


if __name__ == "__main__":
    tests = []
    for t in args.testset:
        tests.extend(get_all_tests(t))
    for t in args.benchmark:
        if '/' in t:
            tests.append(t)
        else:
            tests.append(get_all_tests(t))
    print('Tests:')
    for t in tests:
        print(t)
    print()

    if args.install_ref:
        install_ref(tests, args.install_ref)
    elif args.run:
        print('Running tests\n')
        for t in tests:
            run_test(t)
    else:
        passed = True
        print('Checking tests\n')
        failed = []
        for t in tests:
            res = check_test(t)
            if not res:
                failed.append(t)
            passed = passed and res
        print(f'Final result: {"Passed" if passed else "Failed"}')
        if failed:
            print('Failed test cases:')
            for item in failed:
                print(item)
