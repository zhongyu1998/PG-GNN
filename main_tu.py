import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from utils.process import load_data, separate_data
from models.graphcnn import GraphCNN

torch.backends.cudnn.enabled = False


def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_all = 0
    for _ in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]

        output = model(batch_graph)
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        criterion = nn.CrossEntropyLoss()

        # compute loss
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()

        loss_all += loss.detach().cpu().numpy()

        # report
        pbar.set_description('epoch: %d' % epoch)

    train_loss = loss_all / total_iters
    print("loss training: %f" % train_loss)

    return train_loss


# pass data to model with mini-batch during testing to avoid memory overflow (does not perform back-propagation)
def pass_data_iteratively(model, graphs, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())

    return torch.cat(output, 0)


def evaluate(args, model, device, train_graphs, test_graphs):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    return acc_train, acc_test


def main():
    # Parameters settings
    # Note: Hyper-parameters need to be tuned to obtain the results reported in the paper.
    #       Please refer to our paper for more details about hyper-parameter configurations.
    parser = argparse.ArgumentParser(description='PyTorch implementation of PG-GNN for TU datasets')
    parser.add_argument('--dataset', type=str, default="IMDBBINARY",
                        help='name of dataset (default: IMDBBINARY)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=400,
                        help='maximum number of training epochs (default: 400)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=7,
                        help='random seed for running the experiment (default: 7)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='fold index in 10-fold validation (should be less then 10)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP/RNN EXCLUDING the input one (default: 2)')
    parser.add_argument('--hidden_dim', type=int, default=16,
                        help='number of hidden units (default: 16)')
    parser.add_argument('--final_dropout', type=float, default=0.0,
                        help='dropout ratio after the final layer (default: 0.0)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='pooling for all nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="lstm", choices=["sum", "average", "max", "srn", "gru", "lstm"],
                        help='pooling for neighboring nodes: sum, average, max, srn, gru, or lstm')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be node degrees')
    args = parser.parse_args()

    # set up seeds and gpu device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)

    # 10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim,
                     num_classes, args.final_dropout, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_graphs, optimizer, epoch)
        acc_train, acc_test = evaluate(args, model, device, train_graphs, test_graphs)

        print("accuracy train: %f, test: %f" % (acc_train, acc_test))

        # with open(filename, 'a') as f:
        #     f.write("%f %f %f" % (train_loss, acc_train, acc_test))
        #     f.write("\n")

        scheduler.step()

        print("")


if __name__ == '__main__':
    main()
