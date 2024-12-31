import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import argparse
from datetime import datetime
from omegaconf import OmegaConf
now = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--session_name',     '-sn', default='debug', type=str)
parser.add_argument('--debug_mode',       '-dm', default=False,   action='store_true')
parser.add_argument('--toy_test',         '-tt', default=False,   action='store_true')
parser.add_argument('--multi_gpu',        '-mg', default='0,1',   type=str)

parser.add_argument('--multi_fold',       '-mf', default=1,       type=int)
parser.add_argument('--start_fold',       '-sf', default=1,       type=int)
parser.add_argument('--end_fold',         '-ef', default=5,       type=int)
parser.add_argument('--testing_mode',     '-tm', default=False,   action='store_true')
parser.add_argument('--port_offset',      '-po', default=23,      type=int)

args = parser.parse_args()
SCRIPT_LINE = f'CUDA_VISIBLE_DEVICES={args.multi_gpu} python -W ignore src/train.py'
conf = OmegaConf.load(f'sessions/{args.session_name}.yaml')

if args.debug_mode:
    conf.dev_mode.debugging = True
    args.session_name += '_debug'
    OmegaConf.save(config=conf, f=open(f'sessions/{args.session_name}.yaml', 'w'))
if args.toy_test:
    conf.dev_mode.toy_test = True
    args.session_name += '_toytest'
    OmegaConf.save(config=conf, f=open(f'sessions/{args.session_name}.yaml', 'w'))
if args.testing_mode:
    conf.experiment.test_mode = True
    args.session_name += '_testeval'
    OmegaConf.save(config=conf, f=open(f'sessions/{args.session_name}.yaml', 'w'))

def run_process(fold_num, port_offset):
    conf = OmegaConf.load(f'sessions/{args.session_name}.yaml')
    conf.experiment.fold_num = fold_num
    conf.ddp.port += fold_num
    conf.ddp.port += port_offset
    OmegaConf.save(config=conf, f=open(f'sessions/{args.session_name}_{fold_num}.yaml', 'w'))
    os.system(f'{SCRIPT_LINE} -sn {args.session_name}_{fold_num}')
    
    return fold_num

def multiprocess():
    if args.toy_test:
        print('########################### Toy Test ###########################')
        
    from multiprocessing import Pool
    pool = Pool(args.multi_fold)

    all_folds = [*range(args.start_fold, args.end_fold+1)]
    run_folds_list = [all_folds[start_fold:(start_fold+args.end_fold)]
                      for start_fold in range(0, args.end_fold, args.end_fold)]
    if args.toy_test: run_folds_list = [[1]]

    fold_results_list = []
    for fold in run_folds_list:
        print('Dataset Fold Index: ', fold)
        args_list = [(fold_idx, args.port_offset) for fold_idx in fold]
        fold_results_list.extend(pool.starmap(run_process, args_list))
    pool.close()
    pool.join()


if __name__ == "__main__":
    if args.debug_mode:
        print('########################### Debug Mode ###########################')
        run_process(1, 0)
    else:
        multiprocess()
