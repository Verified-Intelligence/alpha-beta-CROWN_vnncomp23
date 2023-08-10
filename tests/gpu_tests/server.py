"""Server to run GPU tests every other period."""
import argparse
import time
import sys
import os
sys.path.append('../../complete_verifier')
from job_server import JobServer


parser = argparse.ArgumentParser()
parser.add_argument('--interval', type=int, default=-1,
                    help='Interval of running the tests in hours.')
parser.add_argument('--jobs', type=int, default=1,
                    help='Number of jobs in parallel.')
parser.add_argument('--check', action='store_true',
                    help='Check the finished tests only.')
args = parser.parse_args()

DIRS = ['vnncomp21', 'vnncomp22', 'beta_crown']


def main():
    if args.check:
        for dirname in DIRS:
            os.system(f'python test.py -s {dirname}')
        return

    tests = [f'python test.py -s {dirname} --run' for dirname in DIRS]
    print('Tests:')
    print('\n'.join(tests) + '\n')

    server = JobServer(njobs=args.jobs, max_jobs_per_gpu=1)

    while True:
        print('Start running tests')
        server.run_commands(tests)

        # TODO save results

        # TODO notify

        if args.interval == -1:
            break
        else:
            print(f'Tests finished. Sleep for {args.interval} hours.')
            time.sleep(args.interval * 3600)


if __name__ == '__main__':
    main()
