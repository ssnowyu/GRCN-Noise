import torch
import torch.nn as nn
import os, shutil
import numpy as np
import torch_geometric
from torch_geometric.datasets import Planetoid, CoraFull, Amazon, Coauthor
import scipy.sparse as sp

from dataset.cora import CoraDataset


# def load_dataset(name):
#     if name in ["Cora", "CiteSeer", "PubMed"]:
#         dataset = Planetoid(root="./data/" + name, name=name)
#     elif name == "CoraFull":
#         dataset = CoraFull(root="./data/" + name)
#     elif name in ["Computers", "Photo"]:
#         dataset = Amazon(root="./data/" + name, name=name)
#     elif name in ["CS", "Physics"]:
#         dataset = Coauthor(root="./data/" + name, name=name)
#     else:
#         exit("wrong dataset")
#     return dataset


def load_dataset(name, noise_rate: float, sigma: float):
    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = CoraDataset(root="./data/" + name, noise_rate=noise_rate, sigma=sigma)
    elif name == "CoraFull":
        dataset = CoraFull(root="./data/" + name)
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(root="./data/" + name, name=name)
    elif name in ["CS", "Physics"]:
        dataset = Coauthor(root="./data/" + name, name=name)
    else:
        exit("wrong dataset")
    return dataset


# def sample_mask(mask: torch.Tensor, percentage: float):
#     # print(mask)
#     one_indices = torch.nonzero(mask == 1).squeeze()
#     # print(one_indices)
#     num_samples = int(percentage * len(one_indices))
#     selected_indices = torch.randperm(len(one_indices))[:num_samples]
#     new_mask = torch.zeros_like(mask)
#     new_mask[selected_indices] = 1
#     return new_mask

# def add_noise(dataset, noise_rate: float, sigma:float):
#     g = dataset[0]
#     feat = g.x
#     train_mask = g.train_mask
#     val_mask = g.val_mask
#     test_mask = g.test_mask

#     train_noise_mask = sample_mask(train_mask, self.noise_rate).int()
#     val_noise_mask = sample_mask(val_mask, self.noise_rate).int()
#     test_noise_mask = sample_mask(test_mask, self.noise_rate).int()

#     noise_mask = train_noise_mask + val_noise_mask + test_noise_mask

#     num_words = (feat > 0).sum(dim=1)
#     feat = num_words.unsqueeze(1).repeat(1, feat.size(-1)) * feat

#     g.x = feat

#     dataset.


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = mx.sum(1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    r_mat_inv = torch.diag(r_inv)
    mx = torch.mm(r_mat_inv, mx)
    return mx


def create_exp_dir(path, scripts_to_save=None):
    path_split = path.split("/")
    path_i = "."
    for one_path in path_split:
        path_i += "/" + one_path
        if not os.path.exists(path_i):
            os.mkdir(path_i)

    print("Experiment dir : {}".format(path_i))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(model, optimizer, epoch, path, finetune=False):
    if finetune:
        torch.save(model, os.path.join(path, "finetune_model.pt"))
        torch.save(optimizer.state_dict(), os.path.join(path, "finetune_optimizer.pt"))
    else:
        torch.save(model, os.path.join(path, "model.pt"))
        torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    torch.save({"epoch": epoch + 1}, os.path.join(path, "misc.pt"))
