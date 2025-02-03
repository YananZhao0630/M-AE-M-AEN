import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.data import TexasDataset,CornellDataset,WisconsinDataset,SquirrelDataset, ActorDataset, ChameleonDataset
from dgl import AddSelfLoop
import scipy
from scipy import sparse
from scipy.io import savemat
import argparse
import numpy as np
import random

from scipy.sparse import diags

from models.models import GCN
import scipy.sparse as sp
from utils import sparse_mx_to_torch_sparse_tensor
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import time
import sys
import json
import os

def evaluate(g, features, labels, mask, model,adj_low, adj_high,adj_low_unnormalized):
    model.eval()
    with torch.no_grad():
        logits = model(features,adj_low, adj_high,adj_low_unnormalized)
        logits = logits[mask]
        prob_distribution = F.softmax(logits, dim=-1)
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return prob_distribution, correct.item()*1.0 / len(labels)

def train(data, features, labels, train_mask, valid_mask, model, adj_low, adj_high, adj_low_unnormalized, train_epoch, graph_type, prob_lambda, lr, weight_decay, seed):
    # define train/val samples, loss function and optimizer
    # train_mask = masks[0][:,split] # Extacts the train and validation masks from the 'masks' list
    # val_mask = masks[1][:,split] # masks seperate the data into training and validation subsets
    if graph_type == "original":
         g = data
    elif graph_type == "new":
         g = data
         g = generate_ngraph_rand_addedge(g,prob_lambda=prob_lambda, train_mask=train_mask, seed=seed)

    loss_fcn = nn.CrossEntropyLoss() # defines the loss function as cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr= lr, weight_decay=weight_decay)
    # Initializes the Adam optimizer with the model parameters as the optimization variables.
    # training loop
    for epoch in range(train_epoch):
        model.train() # set the model in the training mode
        logits = model(features,adj_low, adj_high,adj_low_unnormalized) # Computes the logits (raw model outputs)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1 == 0:
            predicted_labels, acc = evaluate(g, features, labels, valid_mask, model,adj_low, adj_high,adj_low_unnormalized)
            print(
                "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}".format(
                epoch, loss.item(), acc
                )
            )
    return g


def generate_ngraph_rand_addedge(data, prob_lambda, train_mask, seed):
  labels = data.ndata['label']
  device = labels.device
  labels = labels.cpu().numpy()
  train_nodes = np.where(train_mask.cpu().numpy())[0]
  p_add_edge = prob_lambda  # probability of adding edge

  # random.seed(seed)
  np.random.seed(seed)

  # Create a meshgrid of train nodes
  i, j = np.meshgrid(train_nodes, train_nodes, indexing='ij')
  i = i.flatten()
  j = j.flatten()

  # Filter out self-loops and ensure labels are the same
  mask = (i != j) & (labels[i] == labels[j])

  # Apply the probability mask
  random_prob = np.random.rand(i.size)
  mask = mask & (random_prob < p_add_edge)

  # Filter indices based on mask
  new_edges = (i[mask], j[mask])

  # Coalesce the new edges with existing ones
  new_g = data.clone()
  if new_edges[0].size > 0:
      new_g.add_edges(new_edges[0], new_edges[1])
  # Simplify the graph
      g_simple = dgl.to_simple(new_g.to('cpu'), return_counts='count', writeback_mapping=False)
      print(g_simple)
      g_simple = g_simple.to(device)
      print(f'Edge number increased by: {g_simple.num_edges() - data.num_edges()}')
  #
  return g_simple

def compute_label_distribution(graph, labels, num_classes):
    # Initialize an empty list to store the label distribution for each node
    label_distributions = []

    # Number of classes

    # Iterate through each node in the graph
    for node in range(graph.number_of_nodes()):
        # Get the neighbors of the node
        neighbors = graph.in_edges(node)[0]
        neighbor_labels = labels[neighbors]
        # Count the occurrences of each class in the neighborhood
        label_counts = torch.bincount(neighbor_labels, minlength=num_classes)
        # Normalize to get probabilities
        label_distribution = label_counts.float() / label_counts.sum()
        label_distributions.append(label_distribution)

    return torch.stack(label_distributions)

def normalize_tensor(mx, eqvar=None):
    """
    Row-normalize sparse matrix
    """
    rowsum = torch.sum(mx, 1)
    if eqvar:
        r_inv = torch.pow(rowsum, -1 / eqvar).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    else:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="squirrel"
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="original"
    )
    parser.add_argument(
        "--prob_lambda",
        type=float,
        default=0,
        help= "Probability lambda for new graph generation"
    )
    parser.add_argument(
        "--train_epoch",
        type=int,
        default=100
    )
    parser.add_argument("--hidden", type=int, default=64, help="Number of hidden units.")
    parser.add_argument(
        "--layers", type=int, default=1, help="Number of hidden layers, i.e. network depth"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.6, help="Dropout rate (1 - keep probability)."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "gcn",
            "sgc",
            "graphsage",
            "snowball",
            "gcnII",
            "acmgcn",
            "acmgcnp",
            "acmgcnpp",
            "acmsgc",
            "acmgraphsage",
            "acmsnowball",
            "mlp",
        ],
        help="name of the model",
        default="acmgcn",
    )
    parser.add_argument(
        "--structure_info",
        type=int,
        default=0,
        help="1 for using structure information in acmgcnp, 0 for not",
    )
    parser.add_argument(
        "--variant", type=float, default=0, help="Indicate ACM, GCNII variant models."
    )
    parser.add_argument(
        "--link_init_layers_X", type=int, default=1, help="Number of initial layer"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="learning rate"
    )
    parser.add_argument(
        "--wd", type=float, default=5e-6, help="weight decay"
    )
    parser.add_argument("--seed", type=int, default=0, help="The value of the seed")
    parser.add_argument("--norm",action='store_true', help="Whether to normalize the graph feature")
    parser.add_argument("--feature_low", action='store_true', help="Whether to reduce the graph feature dimension")
    args = parser.parse_args()
    # load and precess dataset
    transform = (
        AddSelfLoop()
    )
    if args.dataset == "texas":
        data_raw = TexasDataset(transform=transform)
    elif args.dataset == "cornell":
        data_raw = CornellDataset(transform=transform)
    elif args.dataset == "wisconsin":
        data_raw = WisconsinDataset(transform=transform)
    elif args.dataset == "squirrel":
        data_raw = SquirrelDataset(transform=transform)
    elif args.dataset == "actor":
        data_raw = ActorDataset(transform=transform)
    elif args.dataset == "chameleon":
        data_raw = ChameleonDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))


    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    data = data_raw[0].to(device)

    if args.norm == True:
        features = data.ndata["feat"]

        if args.feature_low == True:
            reduced_dim = 512
            num_features = features.shape[1]

            selected_indices = torch.randperm(num_features)[:reduced_dim]
            reduced_features = features[:, selected_indices]
            features = reduced_features

        norm = features.norm(p=2, dim=1, keepdim=True)
        features = features/norm
    else:
        features = data.ndata["feat"]
        if args.feature_low == True:
            reduced_dim = 512
            num_features = features.shape[1]

            selected_indices = torch.randperm(num_features)[:reduced_dim]
            reduced_features = features[:, selected_indices]
            features = reduced_features

    labels = data.ndata["label"]
    masks = data.ndata["train_mask"], data.ndata["val_mask"], data.ndata["test_mask"]

    src, dst = data.edges()
    edge_index = torch.stack((src,dst), dim=0)
    n = data.num_nodes()
    adj_low_unnormalized = to_scipy_sparse_matrix(edge_index)
    adj_low_unnormalized = sp.identity(n) + adj_low_unnormalized
    row_sums = np.array(adj_low_unnormalized.sum(axis=1)).flatten()

    # Avoid division by zero
    row_sums[row_sums == 0] = 1

    # Create a diagonal matrix of the inverse of row sums
    d_inv = diags(1.0 / row_sums)

    # Multiply by adj_low_unnormalized to normalize
    adj_low = d_inv.dot(adj_low_unnormalized)
    adj_high = sp.identity(n) - adj_low
    adj_low = sparse_mx_to_torch_sparse_tensor(adj_low).to(device)
    adj_high = sparse_mx_to_torch_sparse_tensor(adj_high).to(device)
    adj_low_unnormalized = sparse_mx_to_torch_sparse_tensor(adj_low_unnormalized).to(device)

    # create ACM-GCN model
    in_size = features.shape[1]
    out_size = data_raw.num_classes

    model = GCN(
        nfeat=features.shape[1],
        nhid=args.hidden,
        nclass=out_size,
        nlayers=args.layers,
        nnodes=features.shape[0],
        dropout=args.dropout,
        model_type=args.model,
        structure_info=args.structure_info,
        variant=args.variant,
        init_layers_X=args.link_init_layers_X,
    ).to(device)

    acc_list = []
    l1_list_mean = []
    l1_list_median = []



    for split in range(10):

        train_mask = masks[0][:, split]
        valid_mask = masks[1][:, split]
        #for time in range(2):
        print("Training...")
        g = train(data, features, labels, train_mask, valid_mask, model,  adj_low, adj_high,adj_low_unnormalized, train_epoch=args.train_epoch, graph_type=args.graph_type, prob_lambda=args.prob_lambda, lr = args.lr, weight_decay=args.wd, seed=args.seed)

        # model testing
        print("Testing....")
        test_mask = masks[2][:, split]
        prob_distribution, acc = evaluate(g, features, labels, test_mask, model,adj_low, adj_high,adj_low_unnormalized)
        acc_list.append(acc*100)
        print("Test accuracy {:.4f}".format(acc))

        num_classes = data_raw.num_classes
        ground_truth_distributions = compute_label_distribution(g, labels, num_classes)[test_mask]

        l1_distances = torch.abs(prob_distribution - ground_truth_distributions).sum(1)

        l1_list_mean.append(np.mean(l1_distances.cpu().numpy()))
        l1_list_median.append(np.median(l1_distances.cpu().numpy()))

    print(acc_list)
    acc_mean = np.mean(acc_list)
    acc_var = np.std(acc_list)
    print("Test accuracy mean {:.4f}".format(acc_mean))
    print("Test accuracy variation {:.4f}".format(acc_var))

    l1_mean = np.mean(l1_list_mean)
    l1_median = np.mean(l1_list_median)
    print("L1 distances mean for all nodes:", l1_mean)
    print("L1 distances median among all nodes:", l1_median)

