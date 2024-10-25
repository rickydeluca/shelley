import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from shelley import SHELLEY_SoG, SHELLEY_SoN
from shelley.data.dataset import SemiSyntheticDataset
from shelley.data.utils import dict_to_perm_mat, move_tensors_to_device
from shelley.evaluation.matchers import greedy_match
from shelley.evaluation.metrics import compute_accuracy


def parse_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description='Read the test configuration.')

    parser.add_argument('-e', '--exp', type=str, required=True, help="Path to the experiment configuration (YAML file).")
    parser.add_argument('--res_dir', type=str, default='results/', help="Path to the directory where to save the test results. Default: results/")
    
    return parser.parse_args()


def read_config_file(yaml_file):
    """
    Read the yaml configuration file and return it
    as an `EasyDict` dictionary.
    """
    with open(yaml_file) as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    return cfg

if __name__ == '__main__':
    # Read configuration file
    args = parse_args()
    cfg = read_config_file(args.exp)

    # Set reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(cfg.SEED)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() and cfg.DEVICE == 'cuda' else 'cpu')
    
    # Init result file
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
    res_file = f'{args.res_dir}/{cfg.NAME}_{cfg.DATA.NAME}_{cfg.TRAIN_MODE}.csv'

    header = ['model', 'data', 'size', 'train_ratio', 'noise_add', 'noise_rm', 'avg_acc', 'std_acc', 'avg_time']
    with open(res_file, 'w') as rf:
        csv_writer = csv.DictWriter(rf, fieldnames=header)
        csv_writer.writeheader()
    
    # Run tests
    noise_types = ['add', 'rm']
    noise_probs = cfg.DATA.NOISE_LEVEL
    train_ratios = cfg.DATA.TRAIN_RATIO

    # ----------------------
    #   Split-on-Node Mode
    # ----------------------
    if cfg.TRAIN_MODE.lower() == 'son':
        for train_ratio in train_ratios:
            for noise_type in noise_types:
                for noise_prob in noise_probs:
                    p_add = noise_prob if noise_type == 'add' else 0.0
                    p_rm = noise_prob if noise_type == 'rm' else 0.0

                    dataset = SemiSyntheticDataset(
                        source_path=cfg.DATA.SOURCE_NET,
                        size=cfg.DATA.SIZE,
                        permute=cfg.DATA.PERMUTE,
                        p_add=p_add,
                        p_rm=p_rm,
                        train_ratio=train_ratio,
                        val_ratio=0,
                        seed=cfg.SEED
                    )

                    dataloader = DataLoader(dataset, shuffle=True)

                    matching_accs = []
                    comp_times = []
                    best_epochs = []

                    for pair_id, pair_dict in enumerate(dataloader):
                        # Init model
                        model = SHELLEY_SoN(cfg)

                        # Move to device
                        pair_dict = move_tensors_to_device(pair_dict, device)
                        try:
                            model = model.to(device)
                        except:
                            pass

                        # Predict alignment
                        start_time = time.time()
                        S, best_epoch = model(pair_dict)
                        elapsed_time = time.time() - start_time

                        # Compute matching accuracy
                        P = greedy_match(S)
                        gt_test = dict_to_perm_mat(pair_dict['gt_test'], pair_dict['graph_pair'][0].num_nodes, pair_dict['graph_pair'][1].num_nodes).detach().cpu().numpy()
                        acc = compute_accuracy(P, gt_test)

                        print(f"\n\nPair {pair_id+1}, Accuracy: {acc}")

                        matching_accs.append(acc)
                        best_epochs.append(best_epoch)
                        comp_times.append(elapsed_time)

                    # Average metrics
                    avg_acc = np.mean(matching_accs)
                    std_acc = np.std(matching_accs)
                    avg_time = np.mean(comp_times)
                    avg_best_epoch = np.mean(best_epochs)

                    # Write results
                    out_data = [{
                        'model': cfg.NAME,
                        'data': cfg.DATA.NAME,
                        'size': len(dataset),
                        'train_ratio': train_ratio,
                        'noise_add': p_add,
                        'noise_rm': p_rm,
                        'avg_acc': avg_acc,
                        'std_acc': std_acc,
                        'avg_time': avg_time
                    }]

                    with open(res_file, 'a', newline='') as rf:
                        csv_writer = csv.DictWriter(rf, fieldnames=header)
                        csv_writer.writerows(out_data)
    
    # ------------------------
    #   Split-on-Graphs Mode
    # ------------------------
    else:
        for noise_type in noise_types:
            for noise_prob in noise_probs:
                p_add = noise_prob if noise_type == 'add' else 0.0
                p_rm = noise_prob if noise_type == 'rm' else 0.0
                
                # Init model, once and for all!
                model = SHELLEY_SoG(cfg)
                model.to(device)

                # Prepare datasets
                dataset = SemiSyntheticDataset(
                    source_path=cfg.DATA.SOURCE_NET,
                    size=cfg.DATA.SIZE,
                    permute=cfg.DATA.PERMUTE,
                    p_add=p_add,
                    p_rm=p_rm,
                    train_ratio=cfg.DATA.TRAIN_RATIO,
                    val_ratio=0,
                    seed=cfg.SEED
                )

                train_dataset, val_dataset, test_dataset = random_split(dataset, [cfg.DATA.TRAIN_RATIO, cfg.DATA.VAL_RATIO, 1 - cfg.DATA.TRAIN_RATIO - cfg.DATA.VAL_RATIO])

                # Dataloaders
                train_loader = DataLoader(train_dataset, shuffle=True)
                val_loader = DataLoader(val_dataset, shuffle=True)
                test_loader = DataLoader(test_dataset, shuffle=False)

                # Train model
                start_time = time.time()
                model.train_eval(train_loader, val_loader=val_loader, device=device, verbose=False)
                training_time = time.time() - start_time

                # Test
                avg_acc, std_acc = model.evaluate(test_loader, use_acc=True)

                # Write results
                out_data = [{
                    'model': cfg.NAME,
                    'data': cfg.DATA.NAME,
                    'size': len(dataset),
                    'train_ratio': cfg.DATA.TRAIN_RATIO,
                    'noise_add': p_add,
                    'noise_rm': p_rm,
                    'avg_acc': avg_acc,
                    'std_acc': std_acc,
                    'avg_time': training_time
                }]

                with open(res_file, 'a', newline='') as rf:
                    csv_writer = csv.DictWriter(rf, fieldnames=header)
                    csv_writer.writerows(out_data)
