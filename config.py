import os
import random
import torch
import numpy

def set_seed(seed = -1):
    if seed == -1:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    
    debug = True
    dumpfile_uniqueid = ''
    seed = 2000
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.abspath(__file__)[:-10] # dont use os.getcwd()
    checkpoint_dir = root_dir + '/exp/snapshots'

    dataset = 'porto'
    dataset_prefix = ''
    dataset_file = ''
    dataset_cell_file = ''
    dataset_embs_file = ''

    min_lon = 0.0
    min_lat = 0.0
    max_lon = 0.0
    max_lat = 0.0
    max_traj_len = 200
    min_traj_len = 20
    cell_size = 100.0
    cellspace_buffer = 500.0

    #===========TrajCL=============
    trajcl_batch_size = 128 
    cell_embedding_dim = 256
    seq_embedding_dim = 256
    moco_proj_dim =  seq_embedding_dim // 2
    moco_nqueue = 2048 
    moco_temperature = 0.05

    trajcl_training_epochs = 20
    trajcl_training_bad_patience = 5
    trajcl_training_lr = 0.001
    trajcl_training_lr_degrade_gamma = 0.5
    trajcl_training_lr_degrade_step = 5
    trajcl_aug1 = 'mask'
    trajcl_aug2 = 'subset'
    trajcl_local_mask_sidelen = cell_size * 11
    
    trans_attention_head = 4
    trans_attention_dropout = 0.1
    trans_attention_layer = 2
    trans_pos_encoder_dropout = 0.1
    trans_hidden_dim = 2048

    traj_simp_dist = 100
    traj_shift_dist = 200
    traj_mask_ratio = 0.3
    traj_add_ratio = 0.3
    traj_subset_ratio = 0.7 # preserved ratio

    test_exp1_lcss_edr_epsilon = 0.25 # normalized


    #===========trajsimi=============
    trajsimi_encoder_name = 'TrajCL'
    trajsimi_encoder_mode = 'finetune_all'
    trajsimi_measure_fn_name = 'edwp'

    trajsimi_batch_size = 128
    trajsimi_epoch = 30
    trajsimi_training_bad_patience = 10
    trajsimi_learning_rate = 0.0001
    trajsimi_learning_weight_decay = 0.0001
    trajsimi_finetune_lr_rescale = 0.5


    @classmethod
    def update(cls, dic: dict):
        for k, v in dic.items():
            if k in cls.__dict__:
                assert type(getattr(Config, k)) == type(v)
            setattr(Config, k, v)
        cls.post_value_updates()


    @classmethod
    def post_value_updates(cls):
        if 'porto' == cls.dataset:
            cls.dataset_prefix = 'porto_20200'
            cls.min_lon = -8.7005
            cls.min_lat = 41.1001
            cls.max_lon = -8.5192
            cls.max_lat = 41.2086

        elif 'Bangkok' == cls.dataset:
            cls.dataset_prefix = 'Bangkok_100'
            cls.min_lon = -8.7005
            cls.min_lat = 41.1001
            cls.max_lon = -8.5192
            cls.max_lat = 41.2086
        else:
            pass
        
        cls.dataset_file = cls.root_dir + '/data/' + cls.dataset_prefix
        cls.dataset_cell_file = cls.dataset_file + '_cell' + str(int(cls.cell_size)) + '_cellspace.pkl'
        cls.dataset_embs_file = cls.dataset_file + '_cell' + str(int(cls.cell_size)) + '_embdim' + str(cls.cell_embedding_dim) + '_embs.pkl'
        set_seed(cls.seed)

        cls.moco_proj_dim =  cls.seq_embedding_dim // 2

    @classmethod
    def to_str(cls): # __str__, self
        dic = cls.__dict__.copy()
        lst = list(filter( \
                        lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                        dic.items() \
                        ))
        return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])
