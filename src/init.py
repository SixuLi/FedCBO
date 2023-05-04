import numpy as np
import os
import logging
import shutil
import torch



def make_dirs(dirname, rm=False):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif rm:
        logging.info('Rm and mkdir {}'.format(dirname))
        shutil.rmtree(dirname)
        os.makedirs(dirname)

class Init:
    def __init__(self, args):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.output_path = os.path.join(args.result_path, args.experiment_name)

        if not args.load_checkpoint:
            make_dirs(args.result_path)
            make_dirs(args.data_path)
            make_dirs(self.output_path)
            args_state = {k: v for k, v in args._get_kwargs()}
            with open(os.path.join(self.output_path, 'result.txt'), 'w') as f:
                print(args_state, file=f)
            with open(os.path.join(self.output_path, 'check_state.txt'), 'w') as f:
                print(args_state, file=f)
