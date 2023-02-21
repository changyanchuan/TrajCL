import math
import logging
import random
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from config import Config as Config
from utils import tool_funcs
from utils.data_loader import read_trajsimi_simi_dataset, read_trajsimi_traj_dataset
from utils.traj import merc2cell2, generate_spatial_features


# distance regression
class TrajSimiRegression(nn.Module):
    def __init__(self, nin):
        super(TrajSimiRegression, self).__init__()
        self.enc = nn.Sequential(nn.Linear(nin, nin),
                                nn.ReLU(),
                                nn.Linear(nin, nin))

    def forward(self, trajs):
        # trajs: [batch_size, emb_size]
        return F.normalize(self.enc(trajs), dim=1) #[batch_size, emb_size]


class TrajSimi:
    def __init__(self, encoder):
        super(TrajSimi, self).__init__()

        self.trajsimiregression = None
        self.encoder = encoder
        self.dic_datasets = TrajSimi.load_trajsimi_dataset()

        self.checkpoint_filepath = '{}/{}_trajsimi_{}_{}_best{}.pt' \
                                    .format(Config.checkpoint_dir, Config.dataset_prefix, \
                                            Config.trajsimi_encoder_name, \
                                            Config.trajsimi_measure_fn_name, \
                                            Config.dumpfile_uniqueid)
        
        self.cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))
        self.cellembs = pickle.load(open(Config.dataset_embs_file, 'rb')).to(Config.device) # tensor


    def train(self):
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("train_trajsimi start.@={:.3f}".format(training_starttime))

        _seq_embedding_dim = Config.seq_embedding_dim

        self.trajsimiregression = TrajSimiRegression(_seq_embedding_dim)
        self.trajsimiregression.to(Config.device)
        self.criterion = nn.MSELoss()
        self.criterion.to(Config.device)
        
        optimizer = torch.optim.Adam( [ \
                        {'params': self.trajsimiregression.parameters(), \
                            'lr': Config.trajsimi_learning_rate, \
                            'weight_decay': Config.trajsimi_learning_weight_decay}, \
                        {'params': self.encoder.clmodel.encoder_q.parameters(), \
                            'lr': Config.trajsimi_learning_rate} ] )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.5)
        
        best_epoch = 0
        best_hr_eval = 0
        bad_counter = 0
        bad_patience = Config.trajsimi_training_bad_patience

        for i_ep in range(Config.trajsimi_epoch):
            _time_ep = time.time()
            train_losses = []
            train_gpus = []
            train_rams = []

            self.trajsimiregression.train()
            self.encoder.train()

            for i_batch, batch in enumerate( self.trajsimi_dataset_generator_pairs_batchi() ):
                _time_batch = time.time()
                optimizer.zero_grad()

                trajs_emb, trajs_emb_p, trajs_len, sub_simi = batch
                embs = self.encoder.interpret(trajs_emb, trajs_emb_p, trajs_len)
                outs = self.trajsimiregression(embs)

                pred_l1_simi = torch.cdist(outs, outs, 1)
                pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
                truth_l1_simi = sub_simi[torch.triu(torch.ones(sub_simi.shape), diagonal = 1) == 1]
                loss_train = self.criterion(pred_l1_simi, truth_l1_simi)

                loss_train.backward()
                optimizer.step()
                train_losses.append(loss_train.item())
                train_gpus.append(tool_funcs.GPUInfo.mem()[0])
                train_rams.append(tool_funcs.RAMInfo.mem())

                if i_batch % 200 == 0 and i_batch:
                    logging.debug("training. ep-batch={}-{}, train_loss={:.4f}, @={:.3f}, gpu={}, ram={}" \
                                .format(i_ep, i_batch, loss_train.item(), 
                                        time.time()-_time_batch, tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))

            scheduler.step() # decay before optimizer when pytorch < 1.1

            # i_ep
            logging.info("training. i_ep={}, loss={:.4f}, @={:.3f}" \
                        .format(i_ep, tool_funcs.mean(train_losses), time.time()-_time_ep))
            
            eval_metrics = self.test(dataset_type = 'eval')
            logging.info("eval.     i_ep={}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f}, gpu={}, ram={}".format(i_ep, *eval_metrics))
            
            hr_eval_ep = eval_metrics[1]
            training_gpu_usage = tool_funcs.mean(train_gpus)
            training_ram_usage = tool_funcs.mean(train_rams)

            # early stopping
            if  hr_eval_ep > best_hr_eval:
                best_epoch = i_ep
                best_hr_eval = hr_eval_ep
                bad_counter = 0

                torch.save({"encoder_q" : self.encoder.clmodel.encoder_q.state_dict(),
                            "trajsimi": self.trajsimiregression.state_dict()}, 
                            self.checkpoint_filepath)
            else:
                bad_counter += 1

            if bad_counter == bad_patience or i_ep + 1 == Config.trajsimi_epoch:
                training_endtime = time.time()
                logging.info("training end. @={:.3f}, best_epoch={}, best_hr_eval={:.4f}" \
                            .format(training_endtime - training_starttime, best_epoch, best_hr_eval))
                break
            
        # test
        checkpoint = torch.load(self.checkpoint_filepath)
        self.trajsimiregression.load_state_dict(checkpoint['trajsimi'])
        self.trajsimiregression.to(Config.device)
        self.trajsimiregression.eval()
        self.encoder.clmodel.encoder_q.load_state_dict(checkpoint['encoder_q'])

        test_starttime = time.time()
        test_metrics = self.test(dataset_type = 'test')
        test_endtime = time.time()
        logging.info("test. @={:.3f}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f}, gpu={}, ram={}".format( \
                    test_endtime - test_starttime, *test_metrics))

        return {'task_train_time': training_endtime - training_starttime, \
                'task_train_gpu': training_gpu_usage, \
                'task_train_ram': training_ram_usage, \
                'task_test_time': test_endtime - test_starttime, \
                'task_test_gpu': test_metrics[4], \
                'task_test_ram': test_metrics[5], \
                'hr5':test_metrics[1], 'hr20':test_metrics[2], 'hr20in5':test_metrics[3]}


    @torch.no_grad()
    def test(self, dataset_type):
        # prepare dataset
        if dataset_type == 'eval':
            datasets_simi, max_distance = self.dic_datasets['evals_simi'], self.dic_datasets['max_distance']
            datasets = self.dic_datasets['evals_traj']

        elif dataset_type == 'test':
            datasets_simi, max_distance = self.dic_datasets['tests_simi'], self.dic_datasets['max_distance']
            datasets = self.dic_datasets['tests_traj']

        self.trajsimiregression.eval()
        self.encoder.eval()

        datasets_simi = torch.tensor(datasets_simi, device = Config.device, dtype = torch.float)
        datasets_simi = (datasets_simi + datasets_simi.T) / max_distance
        traj_outs = []

        # get traj embeddings 
        for i_batch, batch in enumerate(self.trajsimi_dataset_generator_single_batchi(datasets)):
            trajs_emb, trajs_emb_p, trajs_len = batch
            embs = self.encoder.interpret(trajs_emb, trajs_emb_p, trajs_len)
            outs = self.trajsimiregression(embs)
            traj_outs.append(outs)
        
        # calculate similarity
        traj_outs = torch.cat(traj_outs)
        pred_l1_simi = torch.cdist(traj_outs, traj_outs, 1)
        truth_l1_simi = datasets_simi
        pred_l1_simi_seq = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
        truth_l1_simi_seq = truth_l1_simi[torch.triu(torch.ones(truth_l1_simi.shape), diagonal = 1) == 1]

        # metrics
        loss = self.criterion(pred_l1_simi_seq, truth_l1_simi_seq)
        hrA = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 5, 5)
        hrB = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 20, 20)
        hrBinA = TrajSimi.hitting_ratio(pred_l1_simi, truth_l1_simi, 20, 5)
        gpu = tool_funcs.GPUInfo.mem()[0]
        ram = tool_funcs.RAMInfo.mem()

        return loss.item(), hrA, hrB, hrBinA, gpu, ram

  
    # data generator - for test
    @torch.no_grad()
    def trajsimi_dataset_generator_single_batchi(self, datasets):
        cur_index = 0
        len_datasets = len(datasets)

        while cur_index < len_datasets:
            end_index = cur_index + Config.trajsimi_batch_size \
                                if cur_index + Config.trajsimi_batch_size < len_datasets \
                                else len_datasets

            trajs = [datasets[d_idx] for d_idx in range(cur_index, end_index)]

            trajs_cell, trajs_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs])
            trajs_emb_p = [torch.tensor(generate_spatial_features(t, self.cellspace)) for t in trajs_p]
            trajs_emb_p = pad_sequence(trajs_emb_p, batch_first = False).to(Config.device)

            trajs_emb_cell = [self.cellembs[list(t)] for t in trajs_cell]
            trajs_emb_cell = pad_sequence(trajs_emb_cell, batch_first = False).to(Config.device) # [seq_len, batch_size, emb_dim]
            
            trajs_len = torch.tensor(list(map(len, trajs_cell)), dtype = torch.long, device = Config.device)

            yield trajs_emb_cell, trajs_emb_p, trajs_len
            cur_index = end_index


    # pair-wise batchy data generator - for training
    def trajsimi_dataset_generator_pairs_batchi(self):
        datasets_simi, max_distance = self.dic_datasets['trains_simi'], \
                                                self.dic_datasets['max_distance']
        datasets = self.dic_datasets['trains_traj']
        len_datasets = len(datasets)
        datasets_simi = torch.tensor(datasets_simi, device = Config.device, dtype = torch.float)
        datasets_simi = (datasets_simi + datasets_simi.T) / max_distance
        
        count_i = 0
        batch_size = len_datasets if len_datasets < Config.trajsimi_batch_size else Config.trajsimi_batch_size
        counts = math.ceil( (len_datasets / batch_size)**2 )

        while count_i < counts:
            dataset_idxs_sample = random.sample(range(len_datasets), k = batch_size)
            # dataset_idxs_sample.sort(key = lambda idx: len(datasets[idx][1]), reverse = True) # len descending order
            sub_simi = datasets_simi[dataset_idxs_sample][:,dataset_idxs_sample]

            trajs = [datasets[d_idx] for d_idx in dataset_idxs_sample]

            trajs_cell, trajs_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs])
            trajs_emb_p = [torch.tensor(generate_spatial_features(t, self.cellspace)) for t in trajs_p]
            trajs_emb_p = pad_sequence(trajs_emb_p, batch_first = False).to(Config.device)

            trajs_emb_cell = [self.cellembs[list(t)] for t in trajs_cell]
            trajs_emb_cell = pad_sequence(trajs_emb_cell, batch_first = False).to(Config.device) # [seq_len, batch_size, emb_dim]
            
            trajs_len = torch.tensor(list(map(len, trajs_cell)), dtype = torch.long, device = Config.device)

            yield trajs_emb_cell, trajs_emb_p, trajs_len, sub_simi
            count_i += 1


    @staticmethod
    def hitting_ratio(preds: torch.Tensor, truths: torch.Tensor, pred_topk: int, truth_topk: int):
        # hitting ratio and recall metrics. see NeuTraj paper
        # the overlap percentage of the topk predicted results and the topk ground truth
        # overlap(overlap(preds@pred_topk, truths@truth_topk), truths@truth_topk) / truth_topk

        # preds = [batch_size, class_num], tensor, element indicates the probability
        # truths = [batch_size, class_num], tensor, element indicates the probability
        assert preds.shape == truths.shape and pred_topk < preds.shape[1] and truth_topk < preds.shape[1]

        _, preds_k_idx = torch.topk(preds, pred_topk + 1, dim = 1, largest = False)
        _, truths_k_idx = torch.topk(truths, truth_topk + 1, dim = 1, largest = False)

        preds_k_idx = preds_k_idx.cpu()
        truths_k_idx = truths_k_idx.cpu()

        tp = sum([np.intersect1d(preds_k_idx[i], truths_k_idx[i]).size for i in range(preds_k_idx.shape[0])])
        
        return (tp - preds.shape[0]) / (truth_topk * preds.shape[0])


    @staticmethod
    def load_trajsimi_dataset():
        # read (1) traj dataset for trajsimi, (2) simi matrix dataset for trajsimi
        trajsimi_traj_dataset_file = Config.dataset_file
        trajsimi_simi_dataset_file = '{}_traj_simi_dict_{}.pkl'.format( \
                                    Config.dataset_file, Config.trajsimi_measure_fn_name)

        trains_traj, evals_traj, tests_traj = read_trajsimi_traj_dataset(trajsimi_traj_dataset_file)
        trains_traj, evals_traj, tests_traj = trains_traj.merc_seq.values, evals_traj.merc_seq.values, tests_traj.merc_seq.values
        trains_simi, evals_simi, tests_simi, max_distance = read_trajsimi_simi_dataset(trajsimi_simi_dataset_file)
        
        # trains_traj : [[[lon, lat_in_merc], [], ..], [], ...]
        # trains_simi : list of list
        return {'trains_traj': trains_traj, 'evals_traj': evals_traj, 'tests_traj': tests_traj, \
                'trains_simi': trains_simi, 'evals_simi': evals_simi, 'tests_simi': tests_simi, \
                'max_distance': max_distance}


