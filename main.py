# -*- coding: utf-8 -*-
import argparse
import time
import torch
import os
import numpy as np
import networkx as nx
from torch_cluster import random_walk
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from psg.logger import Logger
from psg.model import BaseModel, adjust_lr
from torch_geometric.utils import negative_sampling, to_networkx, to_undirected
from torch_sparse import coalesce, SparseTensor
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='SAGE')
    parser.add_argument('--predictor', type=str, default='MLP')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss_func', type=str, default='AUC')
    parser.add_argument('--neg_sampler', type=str, default='global')
    parser.add_argument('--data_name', type=str, default='ogbl-ddi')
    parser.add_argument('--data_path', type=str, default='dataset')
    parser.add_argument('--eval_metric', type=str, default='hits')
    parser.add_argument('--walk_start_type', type=str, default='edge')
    parser.add_argument('--res_dir', type=str, default='')
    parser.add_argument('--pretrain_emb', type=str, default='')
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--mlp_num_layers', type=int, default=2)
    parser.add_argument('--emb_hidden_channels', type=int, default=256)
    parser.add_argument('--gnn_hidden_channels', type=int, default=256)
    parser.add_argument('--mlp_hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--grad_clip_norm', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_neg', type=int, default=1)
    parser.add_argument('--walk_length', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--year', type=int, default=-1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_lr_decay', type=str2bool, default=False)
    parser.add_argument('--use_node_feats', type=str2bool, default=False)
    parser.add_argument('--use_coalesce', type=str2bool, default=False)
    parser.add_argument('--train_node_emb', type=str2bool, default=True)
    parser.add_argument('--train_on_subgraph', type=str2bool, default=False)
    parser.add_argument('--use_valedges_as_input', type=str2bool, default=False)
    parser.add_argument('--eval_last_best', type=str2bool, default=False)
    parser.add_argument('--random_walk_augment', type=str2bool, default=False)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--node_emb', type=int, default=256)
    parser.add_argument('--relay_seeds', type=int, default=500)
    parser.add_argument('--relay_samples', type=int, default=200)
    parser.add_argument('--use_forman_ricci', type=str2bool, default=False)
    parser.add_argument('--use_edge_simattr', type=str2bool, default=False)
    parser.add_argument('--use_cluster_label', type=str2bool, default=False)
    args = parser.parse_args()
    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_spd_matrix(G, S, max_spd=5):
    spd_matrix = np.zeros((G.number_of_nodes(), len(S)), dtype=np.float32)
    for i, node_S in enumerate(S):
        for node, length in nx.shortest_path_length(G, source=node_S).items():
            spd_matrix[node, i] = min(length, max_spd)
    return spd_matrix

def mbkmeans_clusters(
    X,
    k,
    mb,
    print_silhouette_values,
):
    """Generate clusters and print Silhouette metrics using MBKmeans

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches.
        print_silhouette_values: Print silhouette values per cluster.

    Returns:
        Trained clustering model and labels based on X.
    """
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    print(f"For n_clusters = {k}")
    print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    print(f"Inertia:{km.inertia_}")

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return km, km.labels_

def main():
    args = argument()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name=args.data_name, root=args.data_path)
    data = dataset[0]

    if hasattr(data, 'num_features'):
        num_node_feats = data.num_features
    else:
        num_node_feats = 0

    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    else:
        num_nodes = data.adj_t.size(0)

    split_edge = dataset.get_edge_split()
    if args.data_name == 'ogbl-collab':
        # only train edges after specific year
        if args.year > 0 and hasattr(data, 'edge_year'):
            selected_year_index = torch.reshape(
                (split_edge['train']['year'] >= args.year).nonzero(as_tuple=False), (-1,))
            split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
            split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
            split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
            train_edge_index = split_edge['train']['edge'].t()
            # create adjacency matrix
            new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
            new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
            data.adj_t = SparseTensor(row=new_edge_index[0],
                                      col=new_edge_index[1],
                                      value=new_edge_weight.to(torch.float32))
            data.edge_index = new_edge_index

        # Use training + validation edges
        if args.use_valedges_as_input:
            val_edge_index = split_edge['valid']['edge'].t()
            val_edge_index = to_undirected(val_edge_index)
            data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
            val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=int)
            data.edge_weight = torch.cat([data.edge_weight, val_edge_weight], 0)

            # edge weight normalization
            # data.adj_t = SparseTensor(row=data.edge_index[0],
            #                           col=data.edge_index[1],
            #                           value=data.edge_weight.to(torch.float32))
            # deg = data.adj_t.sum(dim=1).to(torch.float)
            # deg_inv_sqrt = deg.pow(-0.5)
            # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            # split_edge['train']['weight'] = deg_inv_sqrt[data.edge_index[0]] * data.edge_weight * deg_inv_sqrt[
            #     data.edge_index[1]]


    edge_index = data.edge_index.to(device)

    print("edge_index: ", edge_index.size(-1), edge_index.size())

    emb = torch.nn.Embedding(data.num_nodes, args.node_emb).to(device)
    print("node_feature: ", emb.weight[0].size(-1), emb.weight[0].size())
    emb_ea = torch.nn.Embedding(args.num_samples, args.node_emb).to(device)

    # encode distance information
    np.random.seed(0)
    nx_graph = to_networkx(data, to_undirected=True)
    node_mask = []
    for _ in range(args.num_samples):
        node_mask.append(np.random.choice(args.relay_seeds, size=args.relay_samples, replace=False))
    node_mask = np.array(node_mask)
    node_subset = np.random.choice(nx_graph.number_of_nodes(), size=args.relay_seeds, replace=False)
    spd = get_spd_matrix(G=nx_graph, S=node_subset, max_spd=5)
    spd = torch.Tensor(spd).to(device)

    edge_attr = spd[edge_index, :].mean(0)[:, node_mask].mean(2)

    a_max = torch.max(edge_attr, dim=0, keepdim=True)[0]
    a_min = torch.min(edge_attr, dim=0, keepdim=True)[0]
    edge_attr = (edge_attr - a_min) / (a_max - a_min + 1e-6)

    # cluster_num = 1
    target = torch.zeros(num_nodes)
    if args.data_name == 'ogbl-collab':
        import pandas as pd
        if args.use_forman_ricci:
            from GraphRicciCurvature.FormanRicci import FormanRicci
            G = to_networkx(data, to_undirected=True)
            frc = FormanRicci(G)
            frc.compute_ricci_curvature()
            temp = [frc.G[edge_index[0][x].item()][edge_index[1][x].item()].get('formanCurvature') for x in
                    range(edge_index[0, :].size(dim=0))]
            temp = torch.tensor(temp)
            edge_curvature_attr = torch.unsqueeze(temp, 1).to(device)

            # a_max = torch.max(edge_curvature_attr, dim=0, keepdim=True)[0]
            # a_min = torch.min(edge_curvature_attr, dim=0, keepdim=True)[0]
            # edge_curvature_attr = (a_max - edge_curvature_attr + 1e-6) / (a_max - a_min + 1e-6)
            edge_curvature_attr = torch.sigmoid(edge_curvature_attr)
            edge_curvature_attr = torch.sub(1.0, edge_curvature_attr)
            edge_attr = torch.cat((edge_attr, edge_curvature_attr), 1)
            args.num_samples += 1
            emb_ea = torch.nn.Embedding(args.num_samples, args.node_emb).to(device)

        if args.use_edge_simattr:
            # if hasattr(data, 'x'):
            #     if data.x is not None:
            #         node_features_matrix = data.x.to(torch.float)
            node_features = pd.read_csv('./dataset/ogbl_collab/raw/node-feat.csv.gz', header=None, compression='gzip', error_bad_lines=False)
            node_features_matrix = node_features.values
            node_features_matrix = torch.Tensor(node_features_matrix).to(device)
            src_node_feature_matrix = node_features_matrix[edge_index[0, :], :]
            dst_node_feature_matrix = node_features_matrix[edge_index[1, :], :]
            edge_sim_attr = src_node_feature_matrix * dst_node_feature_matrix
            edge_attr = torch.cat((edge_attr, edge_sim_attr), 1)
            edge_simattr_dim = edge_sim_attr.shape[1]
            emb_ea = torch.nn.Embedding(args.num_samples + edge_simattr_dim, args.node_emb).to(device)

        if args.use_cluster_label:
            # if hasattr(data, 'x'):
            #     if data.x is not None:
            #         node_features_matrix = data.x.numpy().astype(np.float16)
            # clustering, node_labels = mbkmeans_clusters(
            #     X=node_features_matrix,
            #     k=50,
            #     mb=500,
            #     print_silhouette_values=False,
            # )
            node_labels_raw = pd.read_csv('./node_label.csv', header=None, error_bad_lines=False)
            node_labels = node_labels_raw.values
            # cluster_num = np.unique(node_labels).size # torch.unique(node_labels_tensor).size(0)
            target = torch.tensor(node_labels, dtype=torch.long)
            target = target.squeeze(-1)

    if hasattr(data, 'edge_weight'):
        if data.edge_weight is not None:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    print(args)

    # create log file and save args
    log_file_name = 'log_' + args.data_name + '_' + str(int(time.time())) + '.txt'
    log_file = os.path.join(args.res_dir, log_file_name)
    with open(log_file, 'a') as f:
        f.write(str(args) + '\n')

    if hasattr(data, 'x'):
        if data.x is not None:
            data.x = data.x.to(torch.float)

    data = data.to(device)

    model = BaseModel(
        lr=args.lr,
        dropout=args.dropout,
        grad_clip_norm=args.grad_clip_norm,
        gnn_num_layers=args.gnn_num_layers,
        mlp_num_layers=args.mlp_num_layers,
        emb_hidden_channels=args.emb_hidden_channels,
        gnn_hidden_channels=args.gnn_hidden_channels,
        mlp_hidden_channels=args.mlp_hidden_channels,
        num_nodes=num_nodes,
        num_node_feats=num_node_feats,
        gnn_encoder_name=args.encoder,
        predictor_name=args.predictor,
        loss_func=args.loss_func,
        optimizer_name=args.optimizer,
        device=device,
        use_node_feats=args.use_node_feats,
        train_node_emb=args.train_node_emb,
        edge_attr=edge_attr,
        emb_ea=emb_ea,
        pretrain_emb=args.pretrain_emb,
        labels=target
    )

    total_params = sum(p.numel() for param in model.para_list for p in param)
    total_params_print = f'Total number of model parameters is {total_params}'
    print(total_params_print)
    with open(log_file, 'a') as f:
        f.write(total_params_print + '\n')

    evaluator = Evaluator(name=args.data_name)

    if args.eval_metric == 'hits':
        loggers = {
            'Hits@20': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
        }
    elif args.eval_metric == 'mrr':
        loggers = {
            'MRR': Logger(args.runs, args),
        }

    if args.random_walk_augment:
        rw_row, rw_col, _ = data.adj_t.coo()
        if args.walk_start_type == 'edge':
            rw_start = torch.reshape(split_edge['train']['edge'], (-1,)).to(device)
        else:
            rw_start = torch.arange(0, num_nodes, dtype=torch.long).to(device)

    for run in range(args.runs):
        model.param_init()
        start_time = time.time()

        cur_lr = args.lr
        for epoch in range(1, 1 + args.epochs):
            if args.random_walk_augment:
                walk = random_walk(rw_row, rw_col, rw_start, walk_length=args.walk_length)
                pairs = []
                weights = []
                for j in range(args.walk_length):
                    pairs.append(walk[:, [0, j + 1]])
                    weights.append(torch.ones((walk.size(0),), dtype=torch.float) / (j + 1))
                pairs = torch.cat(pairs, dim=0)
                weights = torch.cat(weights, dim=0)
                # remove self-loop edges
                mask = ((pairs[:, 0] - pairs[:, 1]) != 0)
                split_edge['train']['edge'] = torch.masked_select(pairs, mask.view(-1, 1)).view(-1, 2)
                split_edge['train']['weight'] = torch.masked_select(weights, mask)

            loss = model.train(data, split_edge,
                               batch_size=args.batch_size,
                               neg_sampler_name=args.neg_sampler,
                               num_neg=args.num_neg,
                               adj_t=edge_index)

            if epoch % args.eval_steps == 0:
                results = model.test(data, split_edge,
                                     batch_size=args.batch_size,
                                     evaluator=evaluator,
                                     eval_metric=args.eval_metric,
                                     adj_t=edge_index)
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                if epoch % args.log_steps == 0:
                    spent_time = time.time() - start_time
                    for key, result in results.items():
                        valid_res, test_res = result
                        to_print = (f'Run: {run + 1:02d}, '
                                    f'Epoch: {epoch:02d}, '
                                    f'Loss: {loss:.4f}, '
                                    f'Learning Rate: {cur_lr:.4f}, '
                                    f'Valid: {100 * valid_res:.2f}%, '
                                    f'Test: {100 * test_res:.2f}%')
                        print(key)
                        print(to_print)
                        with open(log_file, 'a') as f:
                            print(key, file=f)
                            print(to_print, file=f)
                    print('---')
                    print(
                        f'Training Time Per Epoch: {spent_time / args.eval_steps: .4f} s')
                    print('---')
                    start_time = time.time()

            if args.use_lr_decay:
                cur_lr = adjust_lr(model.optimizer,
                                   epoch / args.epochs,
                                   args.lr)

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run, last_best=args.eval_last_best)
            with open(log_file, 'a') as f:
                print(key, file=f)
                loggers[key].print_statistics(run, f=f, last_best=args.eval_last_best)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(last_best=args.eval_last_best)
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(f=f, last_best=args.eval_last_best)


if __name__ == "__main__":
    main()
