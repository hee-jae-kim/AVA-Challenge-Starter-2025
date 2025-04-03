import numpy as np
import argparse
import os
import sys
import pickle
import csv
from scipy.spatial.distance import pdist

sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_smpl_ours import DatasetSMPL_ours
from models.motion_pred import *
from scipy.spatial.distance import pdist, squareform

import pdb
import json
def get_prediction(data_skel_np, data_text_np, algo, sample_num, num_seeds=1, concat_hist=True):

    data_skel_np = data_skel_np.reshape(data_skel_np.shape[0], data_skel_np.shape[1], -1)
    motion3d = tensor(data_skel_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()

    text_embd = tensor(data_text_np, device=device, dtype=dtype).contiguous()

    X = motion3d[:t_his]

    if algo == 'vae':
        X = X.repeat((1, sample_num * num_seeds, 1))
        text_embd = text_embd.repeat((sample_num * num_seeds, 1))
        Y = models[algo].sample_prior(X, text_embd)
    else: 
        raise ValueError(f"Unsupported algorithm: {algo}.") 
        
    if concat_hist:
        Y = torch.cat((X, Y), dim=0)
    Y = Y.permute(1, 0, 2).contiguous().cpu().numpy()
    if Y.shape[0] > 1:
        Y = Y.reshape(-1, sample_num, Y.shape[-2], Y.shape[-1])
    else:
        Y = Y[None, ...]
        
    return Y

def get_gt(data):
    gt = data.reshape(data.shape[0], data.shape[1], -1)
    return gt[:, t_his:, :]

"""metrics"""
def compute_diversity(pred, *args):
    if pred.shape[0] == 1:
        return 0.0
    dist = pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist.mean().item()
    return diversity

def compute_ade_fde(pred, gt, type, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)
    dist_mean_along_time = dist.mean(axis=1)
    
    min_sample = np.argmin(dist_mean_along_time, axis=0)

    if type == 'ADE':
        return dist_mean_along_time[min_sample]
    elif type == 'FDE':
        return dist[min_sample][-1]
    else:
        Exception ('wrong metric')


def compute_stats():
    
    pred_for_submission = []
    gt_for_submission = []
    
    stats_func = {'Diversity': compute_diversity, 'ADE': compute_ade_fde,
                  'FDE': compute_ade_fde}
    stats_names = list(stats_func.keys())
    stats_meter = {x: {y: AverageMeter() for y in algos} for x in stats_names}

    data_gen = dataset.iter_generator(step=cfg.t_his)
    num_samples = 0
    num_seeds = args.num_seeds
    for i, [data_skel, data_text, seq_name] in enumerate(data_gen):
        num_samples += 1
        gt = get_gt(data_skel)

        for algo in algos:
            pred = get_prediction(data_skel, data_text, algo, sample_num=cfg.nk, num_seeds=num_seeds, concat_hist=False)
            for stats in stats_names:
                val = 0
                for pred_i in pred:
    
                    """ 
                    Save for submission
                    """
                    b, _, _ = pred_i.shape
                    pred_i_w_root = pred_i.reshape(b, cfg.t_pred, cfg.nj -1, 3)
                    root_joint = np.zeros((b, cfg.t_pred, 1, 3))
                    pred_i_w_root = np.concatenate((root_joint, pred_i_w_root), axis=2)  # shape: (b, t_pred, 23, 3)
                    pred_i_w_root = pred_i_w_root[0] # b is 1 during eval
                    pred_i_flat = pred_i_w_root.flatten().tolist()

                    gt_w_root = gt.reshape(b, cfg.t_pred, cfg.nj -1, 3)
                    root_joint = np.zeros((b, cfg.t_pred, 1, 3))
                    gt_w_root = np.concatenate((root_joint, gt_w_root), axis=2)  # shape: (b, t_pred, 23, 3)
                    gt_w_root = gt_w_root[0] # b is 1 during eval
                    gt_flat = gt_w_root.flatten().tolist()

                    pred_for_submission.append({
                        "seq_name": seq_name,
                        "motion": pred_i_flat
                    })
                    
                    gt_for_submission.append({
                        "seq_name": seq_name,
                        "motion": gt_flat
                    })
                                
                    """ 
                    Compute Metrics
                    """                            
                    val += stats_func[stats](pred_i, gt, stats) / num_seeds
                stats_meter[stats][algo].update(val)
        
        print('-' * 80)
        for stats in stats_names:
            str_stats = f'{num_samples:04d} {stats}: ' + ' '.join([f'{x}: {y.val:.4f}({y.avg:.4f})' for x, y in stats_meter[stats].items()])
            print(str_stats)

    logger.info('=' * 80)
    for stats in stats_names:
        str_stats = f'Total {stats}: ' + ' '.join([f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()])
        logger.info(str_stats)
    logger.info('=' * 80)

    with open('%s/stats_%s.csv' % (cfg.result_dir, args.num_seeds), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + algos)
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            new_meter['Metric'] = stats
            writer.writerow(new_meter)

    with open(f'{cfg.result_dir}/Submission_anonymized.json', 'w') as f:
        json.dump(pred_for_submission, f)
    # with open(f'{cfg.result_dir}/GT_anonymized.json', 'w') as f:
    #     json.dump(gt_for_submission, f)
        
if __name__ == '__main__':

    all_algos = ['vae']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='stats')
    parser.add_argument('--data', default='test')
    parser.add_argument('--action', default='all')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=-1)
    for algo in all_algos:
        parser.add_argument('--iter_%s' % algo, type=int, default=None)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if args.gpu_index >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    torch.set_grad_enabled(False)
    cfg = Config(args.cfg)
    logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

    algos = []
    for algo in all_algos:
        iter_algo = 'iter_%s' % algo
        num_algo = 'num_%s_epoch' % algo
        setattr(args, iter_algo, getattr(cfg, num_algo))
        algos.append(algo)
    vis_algos = algos.copy()

    if args.action != 'all':
        args.action = set(args.action.split(','))

    """parameter"""
    nz = cfg.nz
    nk = cfg.nk
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    nj = cfg.nj

    """data"""
    dataset_cls = DatasetSMPL_ours
    dataset = dataset_cls(args.data, t_his, t_pred, actions=args.action, use_vel=cfg.use_vel)
    
    """models"""
    model_generator = {
        'vae': get_vae_model,
    }
    
    models = {}
    for algo in algos:
        models[algo] = model_generator[algo](cfg, dataset.traj_dim)
        model_path = getattr(cfg, f"{algo}_model_path") % getattr(args, f'iter_{algo}')
        print(f'loading {algo} model from checkpoint: {model_path}')
        model_cp = pickle.load(open(model_path, "rb"))
        models[algo].load_state_dict(model_cp['model_dict'])
        models[algo].to(device)
        models[algo].eval()

    if args.mode == 'vis':
        visualize()
    elif args.mode == 'stats':
        compute_stats()
