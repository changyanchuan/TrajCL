import sys
import logging
import argparse
import os
import numpy as np
from config import Config
from utils import tool_funcs
from task.trajsimi import TrajSimi
from model.trajcl import TrajCL

def parse_args():
    # dont set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.
    parser = argparse.ArgumentParser(description = "TrajCL/train_trajsimi.py")
    parser.add_argument('--dumpfile_uniqueid', type = str, help = 'see config.py')
    parser.add_argument('--seed', type = int, help = '')
    parser.add_argument('--dataset', type = str, help = '')
    
    parser.add_argument('--trajsimi_measure_fn_name', type = str, help = '')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


def main():
    enc_name = Config.trajsimi_encoder_name
    fn_name = Config.trajsimi_measure_fn_name
    metrics = tool_funcs.Metrics()

    trajcl = TrajCL()
    trajcl.load_checkpoint()
    trajcl.to(Config.device)
    task = TrajSimi(trajcl)
    tasktrain = task.train()
    pred_l1_simi_np,truth_l1_simi_np,datasets_simi2_np = tasktrain['pred_l1_simi_np'],tasktrain['truth_l1_simi_np'],tasktrain['datasets_simi2_np']
    # ลบ key และ value ออกจาก dictionary
    del tasktrain['pred_l1_simi_np']
    del tasktrain['truth_l1_simi_np']
    del tasktrain['datasets_simi2_np']
    metrics.add(tasktrain)
    # Ensure that the directory exists
    log_dir = os.path.join(Config.root_dir, 'exp', 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Define file paths
    pred_file_path = os.path.join(log_dir, 'pred_l1_simi.npy')
    truth_file_path = os.path.join(log_dir, 'truth_l1_simi.npy')
    datasets_file_path = os.path.join(log_dir, 'datasets_simi2_np')

    # Save the arrays as .npy files in the log folder
    np.save(pred_file_path, pred_l1_simi_np)
    np.save(truth_file_path, truth_l1_simi_np)
    np.save(datasets_file_path, datasets_simi2_np)
    
    logging.info('[EXPFlag]model={},dataset={},fn={},{}'.format( \
                enc_name, Config.dataset_prefix, fn_name, str(metrics)))
    return


# nohup python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name hausdorff &> result &
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

    main()
    
