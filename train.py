import sys
import logging
import argparse

from model.trajcl import TrajCLTrainer
from config import Config
from utils import tool_funcs


def parse_args():
    # dont set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.
    
    parser = argparse.ArgumentParser(description = "TrajCL/train.py")
    parser.add_argument('--dumpfile_uniqueid', type = str, help = 'see config.py')
    parser.add_argument('--seed', type = int, help = '')
    parser.add_argument('--dataset', type = str, help = '')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


# nohup python train.py --dataset porto &> result &
if __name__ == '__main__':
    Config.update(parse_args())
    logging.basicConfig(level = logging.DEBUG if Config.debug else logging.INFO,
            format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
            handlers = [logging.FileHandler(Config.root_dir+'/exp/log/'+tool_funcs.log_file_name(), mode = 'w'), 
                        logging.StreamHandler()]
            )

    logging.info('python ' + ' '.join(sys.argv))
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    trajcl = TrajCLTrainer(Config.trajcl_aug1, Config.trajcl_aug2)
    trajcl.train()
    trajcl.test()
    
    