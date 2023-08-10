### preprocessor-hint: private-file
from datetime import datetime
import os
import re
import argparse
import pickle
from job_server import JobServer

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'gpu'], help='Choose the device to run experiments/')
parser.add_argument('--parallel_device', type=str, default='cuda', choices=['cpu', 'cuda', 'gpu'], help='Choose the device to run experiments/ in parallel')
parser.add_argument('--num_tasks', type=int, required=True, help='number of task to run in parallel.')
parser.add_argument('--set_affinity', action='store_true', help='Set each task to a fixed set of CPUs.')
parser.add_argument('--ncpus', type=int, default=1, help='number of CPUs to use for each task (only valid when --set_affinity is given).')
parser.add_argument('--start_cpuid', type=int, default=0, help='the ID of the first CPU core to use (only valid when --set_affinity is given).')
parser.add_argument('--start', type=int, default=0, help='start index')
parser.add_argument('--end', type=int, default=-1, help='end index')
parser.add_argument('--chunk_size', type=int, default=20, help='number of examples to verify in each job')
parser.add_argument('--log_dir', type=str, default=None, help='log file path. If set to None, it will generate a new folder based on date and time.')
parser.add_argument('--no_confirm', action='store_true', help='no need to confirm before start.')
parser.add_argument('--data_idx_file', default=None, help='use the data idx file to identify the start and end automatically.')
parser.add_argument('--dir_prefix', default='verifier_log', help='specify the prefix of output dir when --log_dir is not specified.')
parser.add_argument('--command_file', type=str, help='Specify a file and this script will run all the commands in the file (one command per line).')
parser.add_argument('--results', '--res', action='store_true', help='Only gather the existing results without running the commands.')
parser.add_argument('--utilization', type=int, default=5)
parser.add_argument('--memory_ratio', type=float, default=0.05)
args, unknown = parser.parse_known_args()

commands = []
result_files = []

if args.command_file:
    # Simply run all the commands in the file (one command per line)
    with open(args.command_file) as file:
        commands = list(file.readlines())
else:
    if args.end == -1:
        if args.data_idx_file is not None:
            with open(args.data_idx_file) as f:
                bnb_ids = re.split('[;|,|\n|\s]+', f.read().strip())
                args.end = len(bnb_ids)
        else:
            args.end = 10000
        print(f'automatically setting --end to {args.end}')

    if args.log_dir is None:
        args.log_dir = datetime.now().strftime(f'{args.dir_prefix}_%Y%m%d_%H%M%S')
    os.makedirs(args.log_dir, exist_ok=True)

    for start in range(args.start, args.end, args.chunk_size):
        end = min(start + args.chunk_size, args.end)
        command = " ".join(unknown + ["--start", f"{start}", "--end", f"{end}"])
        if args.data_idx_file is not None:
            command += f" --data_idx_file {args.data_idx_file}"
        result_file = f'{args.log_dir}/job_{start:04d}_{end:04d}.pkl'
        command = (f"unbuffer python abcrown.py --device {args.device} {command} "
                   f"--results_file {result_file} "
                   f"2>&1 | tee {args.log_dir}/job_{start:04d}_{end:04d}.log")
        commands.append(command)
        result_files.append(result_file)


if not args.results:
    use_gpu = (not args.device == 'cpu') and (not args.parallel_device == 'cpu')
    server = JobServer(njobs=args.num_tasks, max_jobs_per_gpu=1,
                       use_delay=use_gpu, use_gpu=use_gpu,
                       utilization=args.utilization,
                       memory_ratio=args.memory_ratio)
    print(f'Logging to {args.log_dir}')
    print('We will run the following commands:')
    for cmd in commands:
        print(cmd)
    if not args.no_confirm:
        print('Please check commands and log file names, and then press enter to start.')
        input()
    server.run_commands(commands)
    print()

count = {}
results = []
for result_file in result_files:
    # TODO gather timeout?
    with open(result_file, 'rb') as file:
        res = pickle.load(file)
    basename = os.path.basename(result_file)[4:]
    start_idx = int(basename[:basename.find('_')])
    for k, v in res['summary'].items():
        if k not in count:
            count[k] = 0
        count[k] += len(v)
        for item in v:
            results.append((item + start_idx, k))
results = sorted(results)
for item in results:
    print(item[0], item[1])
print('Total:', count)
