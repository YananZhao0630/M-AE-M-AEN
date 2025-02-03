import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.data import TexasDataset,CornellDataset,WisconsinDataset,SquirrelDataset, ActorDataset, ChameleonDataset
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl import AddSelfLoop
from ogb.nodeproppred import DglNodePropPredDataset
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
        # predicted_labels = torch.argmax(logits, dim=-1)
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

def load_ogbn_arxiv():
    dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    g, labels = dataset[0]
    g.ndata['label'] = labels.squeeze()
    g.num_classes = dataset.num_classes

    split_idx = dataset.get_idx_split()
    # Initialize the masks with zeros
    train_mask = np.zeros(g.num_nodes(), dtype=bool)
    valid_mask = np.zeros(g.num_nodes(), dtype=bool)
    test_mask = np.zeros(g.num_nodes(), dtype=bool)

    # Set the corresponding indices to True
    train_mask[split_idx['train']] = True
    valid_mask[split_idx['valid']] = True
    test_mask[split_idx['test']] = True

    # Convert masks to PyTorch tensors and assign to graph
    g.ndata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool)
    g.ndata['val_mask'] = torch.tensor(valid_mask, dtype=torch.bool)
    g.ndata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool)

    g = dgl.add_self_loop(g)

    return g
#def rand_train_test_idx(label, train_prop=.6, valid_prop=.2, ignore_negative=True,seed=None):
#    """ randomly splits label into train/valid/test splits """
#    if ignore_negative:
#        labeled_nodes = torch.where(label != -1)[0]
#    else:
#        labeled_nodes = label

#    if seed is not None:
#        random.seed(seed)
#        np.random.seed(seed)

#    n = labeled_nodes.shape[0]
#    train_num = int(n * train_prop)
#    valid_num = int(n * valid_prop)
#    perm = torch.as_tensor(np.random.permutation(n), dtype=torch.int64)
#    train_indices = perm[:train_num]
#    val_indices = perm[train_num:train_num + valid_num]
#    test_indices = perm[train_num + valid_num:]
#    if not ignore_negative:
#        return train_indices, val_indices, test_indices
#    train_idx = labeled_nodes[train_indices]
#    valid_idx = labeled_nodes[val_indices]
#    test_idx = labeled_nodes[test_indices]
#    return train_idx, valid_idx, test_idx

# def generate_ngraph_rand_addedge(g, prob_lambda, train_mask):
#     labels = g.ndata['label']
#     device = labels.device
#     # train_idx, valid_idx, test_idx = rand_train_test_idx(labels, train_prop=0.6, valid_prop=0.2, ignore_negative=True,seed=None)
#     #train_mask = g.ndata['train_mask'][:,0]
#     # print(train_idx, valid_idx, test_idx)
#     # train_nodes = train_idx.cpu().numpy()
#     train_nodes = np.where(train_mask.cpu().numpy())[0]
#     print(train_nodes.shape)
#
#
#     p_add_edge = prob_lambda  # probability of adding edge
#
#     new_edges = []
#     for i in train_nodes:
#         for j in train_nodes:
#             if i != j and labels[i] == labels[j] and np.random.rand() < p_add_edge:
#                 new_edges.append((i, j))
#
#     new_g = g.clone()
#     if new_edges:
#         new_edges = np.array(new_edges).T
#         new_g.add_edges(new_edges[0], new_edges[1])
#
#     # print(new_edges.shape)
#     g_simple = dgl.to_simple(new_g.to('cpu'), return_counts='count', writeback_mapping=False)
#     print(g_simple)
#     g_simple = g_simple.to(device)
#     print('edge num increased:', g_simple.num_edges()-g.num_edges())
#     return g_simple
# def generate_ngraph_rand_addedge(g, prob_lambda, train_mask):
#     labels = g.ndata['label']
#     device = labels.device
#
#     # Get indices of training nodes
#     # train_nodes = torch.where(train_mask)[0]
#     train_nodes = np.where(train_mask)[0]
#     print(f"Train nodes shape: {train_nodes.shape}")
#
#     p_add_edge = prob_lambda  # probability of adding edge
#
#     # Create a meshgrid of train nodes indices
#     i, j = np.meshgrid(train_nodes, train_nodes, indexing='ij')
#     i = i.flatten()
#     j = j.flatten()
#
#     # Ensure no self loops and labels match
#     mask = (i != j) & (labels[i] == labels[j])
#
#     # Apply the probability mask
#     random_prob = np.random.rand(i.size)
#     mask = mask & (random_prob < p_add_edge)
#
#     new_edges = (i[mask], j[mask])
#
#     # Clone the graph and add edges
#     new_g = g.clone()
#     if new_edges[0].size(0) > 0:
#         new_g.add_edges(new_edges[0], new_edges[1])
#
#     # Simplify the graph
#     g_simple = dgl.to_simple(new_g.to('cpu'), return_counts='count', writeback_mapping=False)
#     print(g_simple)
#     g_simple = g_simple.to(device)
#     print(f'Edge number increased by: {g_simple.num_edges() - g.num_edges()}')
#
#     return g_simple

# def generate_ngraph_rand_addedge(data, prob_lambda, train_mask, seed):
#   labels = data.ndata['label']
#   device = labels.device
#   labels = labels.cpu().numpy()
#   train_nodes = np.where(train_mask.cpu().numpy())[0]
#   p_add_edge = prob_lambda  # probability of adding edge
#
#   # random.seed(seed)
#   np.random.seed(seed)
#
#   # Create a meshgrid of train nodes
#   i, j = np.meshgrid(train_nodes, train_nodes, indexing='ij')
#   i = i.flatten()
#   j = j.flatten()
#
#   # ------------------------------------
#   # num_train_nodes = len(train_nodes)
#   #
#   # # Create indices for train nodes
#   # i = np.repeat(train_nodes, num_train_nodes)
#   # j = np.tile(train_nodes, num_train_nodes)
#
#   # Filter out self-loops and ensure labels are the same
#   mask = (i != j) & (labels[i] == labels[j])
#
#   # Apply the probability mask
#   random_prob = np.random.rand(i.size)
#   mask = mask & (random_prob < p_add_edge)
#
#   # Filter indices based on mask
#   new_edges = (i[mask], j[mask])
#
#   # Coalesce the new edges with existing ones
#   new_g = data.clone()
#   if new_edges[0].size > 0:
#      new_g.add_edges(new_edges[0], new_edges[1])
#   # Simplify the graph
#   g_simple = dgl.to_simple(new_g.to('cpu'), return_counts='count', writeback_mapping=False)
#   print(g_simple)
#   g_simple = g_simple.to(device)
#   print(f'Edge number increased by: {g_simple.num_edges() - data.num_edges()}')
#   #
#   return g_simple


def generate_ngraph_rand_addedge(data, prob_lambda, train_mask, seed):
    labels = data.ndata['label']
    device = labels.device
    labels = labels.cpu().numpy()
    train_nodes = np.where(train_mask.cpu().numpy())[0]
    p_add_edge = prob_lambda  # probability of adding edge

    np.random.seed(seed)

    num_train_nodes = len(train_nodes)

    # Initialize lists to hold edges
    src_list = []
    dst_list = []

    # Iterate over all train nodes
    for idx, u in enumerate(train_nodes):
        # Find all nodes with the same label
        same_label_nodes = train_nodes[labels[train_nodes] == labels[u]]

        # Exclude the node itself
        same_label_nodes = same_label_nodes[same_label_nodes != u]

        # Apply the probability mask
        random_prob = np.random.rand(len(same_label_nodes))
        same_label_nodes = same_label_nodes[random_prob < p_add_edge]

        # Append edges to the lists
        src_list.extend([u] * len(same_label_nodes))
        dst_list.extend(same_label_nodes)

    src = np.array(src_list)
    dst = np.array(dst_list)

    # Coalesce the new edges with existing ones
    new_g = data.clone()
    if src.size > 0:
        new_g.add_edges(src, dst)

    # Simplify the graph
    g_simple = dgl.to_simple(new_g.to('cpu'), return_counts='count', writeback_mapping=False)
    g_simple = g_simple.to(device)
    print(f'Edge number increased by: {g_simple.num_edges() - data.num_edges()}')

    return g_simple


def compute_label_distribution(graph, labels, num_classes):
    # Initialize an empty list to store the label distribution for each node
    label_distributions = []

    # Number of classes

    # Iterate through each node in the graph
    for node in range(graph.number_of_nodes()):
        # Get the neighbors of the node
        neighbors = graph.in_edges(node)[0]
        # neighbors = list(graph.neighbors(node))
        # Include the node itself in the neighbors list
        # neighbors.append(node)  # 加入节点自己
        # Get the labels of the neighbors
        # print(neighbors)
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
        "--dropout", type=float, default=0.06, help="Dropout rate (1 - keep probability)."
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
        "--wd", type=float, default=1e-5, help="weight decay"
    )
    parser.add_argument("--seed", type=int, default=0, help="The value of the seed")
    args = parser.parse_args()
    # load and precess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "texas":
        data_raw = TexasDataset(transform=transform)
        data_raw = data_raw[0]
    elif args.dataset == "cornell":
        data_raw = CornellDataset(transform=transform)
        data_raw = data_raw[0]
    elif args.dataset == "wisconsin":
        data_raw = WisconsinDataset(transform=transform)
        data_raw = data_raw[0]
    elif args.dataset == "squirrel":
        data_raw = SquirrelDataset(transform=transform)
        data_raw = data_raw
    elif args.dataset == "actor":
        data_raw = ActorDataset(transform=transform)
        data_raw = data_raw
    elif args.dataset == "chameleon":
        data_raw = ChameleonDataset(transform=transform)
        data_raw = data_raw
    elif args.dataset == "cora":
        data_raw = CoraGraphDataset(transform=transform)
        data_raw = data_raw
    elif args.dataset == "citeseer":
        data_raw = CiteseerGraphDataset(transform=transform)
        data_raw = data_raw
    elif args.dataset == "pubmed":
        data_raw = PubmedGraphDataset(transform=transform)
        data_raw = data_raw
    elif args.dataset == "ogbn_arkiv":
        data_raw = load_ogbn_arxiv()
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))


    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    data = data_raw[0].to(device)
    #  g_new = g_new.int().to(device)
    if args.dataset == "actor" or args.dataset == "chameleon" or args.dataset == "squirrel":
        features = data.ndata["feat"]
        features = features + 0.1 / (features.shape[1]) * torch.ones_like(data.ndata["feat"])
    else:
        features = data.ndata["feat"]
    # features = data.ndata["feat"]
    labels = data.ndata["label"]
    masks = data.ndata["train_mask"], data.ndata["val_mask"], data.ndata["test_mask"]
    # masks_com = g.ndata("train_mask","val_mask","test_mask")
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
    # print(edge_index)
    # train_idx, valid_idx, test_idx = rand_train_test_idx(labels, train_prop=0.6, valid_prop=0.2, ignore_negative=True, seed=None)
    # create GCN model
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
   # logits = model(features, adj_low, adj_high, adj_low_unnormalized)
    acc_list = []
    l1_list_mean = []
    l1_list_median = []


    # timestr = time.strftime("%Y%m%d-%H%M%S")
    # filename = ("log/"+ "ACM_GCN/"+ str(args.dataset) + str(args.graph_type) + str(args.prob_lambda)+ timestr + ".txt")
    # command_args = " ".join(sys.argv)
    # if not os.path.exists(filename):
    #     # 如果文件不存在，首次写入时包含命令
    #     mode = 'w'  # 写入模式，会创建文件
    # else:
    #     # 如果文件已存在，不重复写入命令
    #     mode = 'a'  # 追加模式
    # with open(filename, mode) as f:
    #     json.dump(command_args, f)
    #     f.write("\n")



    for split in range(10):

        train_mask = masks[0][:, split]
        valid_mask = masks[1][:, split]
        #for time in range(2):
        print("Training...")
        g = train(data, features, labels, train_mask, valid_mask, model,  adj_low, adj_high,adj_low_unnormalized, train_epoch=args.train_epoch, graph_type=args.graph_type, prob_lambda=args.prob_lambda, lr = args.lr, weight_decay=args.wd, seed=args.seed)

        # model testing
        print("Testing....")
        test_mask = masks[2][:,split]
        prob_distribution, acc = evaluate(g, features, labels, test_mask, model,adj_low, adj_high,adj_low_unnormalized)
        acc_list.append(acc*100)
        print("Test accuracy {:.4f}".format(acc))

        num_classes = data_raw.num_classes
        ground_truth_distributions = compute_label_distribution(g, labels, num_classes)[test_mask]
        # calculate distance
        l1_distances = torch.abs(prob_distribution - ground_truth_distributions).sum(1)
        # calculate the mean and median of distance
        l1_list_mean.append(np.mean(l1_distances.cpu().numpy()))
        l1_list_median.append(np.median(l1_distances.cpu().numpy()))

    print(acc_list)
    acc_mean = np.mean(acc_list)
    acc_var = np.std(acc_list)
    print("Test accuracy mean {:.4f}".format(acc_mean))
    print("Test accuracy variation {:.4f}".format(acc_var))

    # train(g, features, labels, masks, model, train_epoch=args.train_epoch)
    # predicted_labels, acc = evaluate(g, features, labels, masks[2], model)

    # predicted_distributions = compute_label_distribution(g, predicted_labels,num_classes)


    # print("L1 distances for all nodes:", l1_distances)
    l1_mean = np.mean(l1_list_mean)
    l1_median = np.mean(l1_list_median)
    print("L1 distances mean for all nodes:", l1_mean)
    print("L1 distances median among all nodes:", l1_median)

    # with open(filename, 'a') as f:
    #     f.write("Test accuracy mean: ")
    #     json.dump(str(acc_mean), f)
    #     f.write("\n")
    #
    #     f.write("Test accuracy variation: ")
    #     json.dump(str(acc_var), f)
    #     f.write("\n")
    #
    #     f.write("Training epoch: ")
    #     json.dump(str(args.train_epoch), f)
    #     f.write("\n")
    #
    #     f.write("graph_type: ")
    #     json.dump(str(args.graph_type), f)
    #     f.write("\n")
    #
    #     f.write("prob_lambda: ")
    #     json.dump(str(args.prob_lambda), f)
    #     f.write("\n")
    #
    #     f.write("L1 distance mean for all nodes:")
    #     json.dump(str(l1_mean), f)
    #     f.write("\n")
    #     f.write("L1 distances median among all nodes:")
    #     json.dump(str(l1_median), f)
    #     f.write("\n")