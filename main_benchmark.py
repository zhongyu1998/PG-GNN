import argparse
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from utils.process import count_node_type, load_cvdata, load_zincdata
from models.graphcnn import GraphCNN

torch.backends.cudnn.enabled = False


def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    pbar = tqdm(range(0, len(train_graphs), args.batch_size), unit='batch')

    loss_all = 0
    for iteration, _ in enumerate(pbar):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]

        if args.dataset == "ZINC":
            output = model(batch_graph)
            labels = torch.FloatTensor([graph.label for graph in batch_graph]).view_as(output).to(device)
            criterion = nn.MSELoss()
        if args.dataset == "MNIST":
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

    train_loss = loss_all / (iteration + 1)
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


def evaluate(args, model, device, graphs):
    model.eval()

    if args.dataset == "ZINC":
        output = pass_data_iteratively(model, graphs)
        labels = torch.FloatTensor([graph.label for graph in graphs]).view_as(output).to(device)
        mae = nn.L1Loss(reduction='mean')
        result = mae(output, labels).cpu().item()
    if args.dataset == "MNIST":
        output = pass_data_iteratively(model, graphs)
        pred = output.max(1, keepdim=True)[1]
        labels = torch.LongTensor([graph.label for graph in graphs]).view_as(pred).to(device)
        correct = pred.eq(labels).sum().cpu().item()
        result = correct / float(len(graphs))

    return result


def main():
    # Parameters settings
    # Note: Hyper-parameters need to be tuned to obtain the results reported in the paper.
    #       Please refer to our paper for more details about hyper-parameter configurations.
    parser = argparse.ArgumentParser(description='PyTorch implementation of PG-GNN for Benchmarking datasets')
    parser.add_argument('--dataset', type=str, default="ZINC",
                        help='name of dataset: MNIST or ZINC (default: ZINC)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='maximum number of training epochs (default: 500)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='reduce factor of learning rate (default: 0.5)')
    parser.add_argument('--lr_patience', type=int, default=25,
                        help='decay rate patience of learning rate (default: 25)')
    parser.add_argument('--lr_limit', type=float, default=5e-6,
                        help='minimum learning rate, stop training once it is reached (default: 5e-6)')
    parser.add_argument('--seed', type=int, default=9,
                        help='random seed for running the experiment (default: 9)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP/RNN EXCLUDING the input one (default: 2)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='number of hidden units (default: 128)')
    parser.add_argument('--final_dropout', type=float, default=0.0,
                        help='dropout ratio after the final layer (default: 0.0)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='pooling for all nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="lstm", choices=["sum", "average", "max", "srn", "gru", "lstm"],
                        help='pooling for neighboring nodes: sum, average, max, srn, gru, or lstm')
    args = parser.parse_args()

    # set up seeds and gpu device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.dataset == "MNIST":
        mode = 'max'
        with open('data/superpixels/%s.pkl' % args.dataset, 'rb') as f:
            trainset, valset, testset = pickle.load(f)
        train_graphs, num_classes = load_cvdata(trainset, state='train')
        val_graphs, _ = load_cvdata(valset, state='val')
        test_graphs, _ = load_cvdata(testset, state='test')
    if args.dataset == "ZINC":
        mode = 'min'
        with open('data/molecules/%s.pkl' % args.dataset, 'rb') as f:
            trainset, valset, testset, num_atom_type, num_bond_type = pickle.load(f)
        num_classes = 1
        num_node_type = count_node_type(trainset, valset, testset)
        train_graphs = load_zincdata(trainset, num_node_type, state='train')
        val_graphs = load_zincdata(valset, num_node_type, state='val')
        test_graphs = load_zincdata(testset, num_node_type, state='test')

    model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim,
                     num_classes, args.final_dropout, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode,
                                                     factor=args.lr_factor,
                                                     patience=args.lr_patience,
                                                     verbose=True)

    train_curve = []
    val_curve = []
    test_curve = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_graphs, optimizer, epoch)
        result_train = evaluate(args, model, device, train_graphs)
        result_val = evaluate(args, model, device, val_graphs)
        result_test = evaluate(args, model, device, test_graphs)

        print("result train: %f, val: %f, test: %f" % (result_train, result_val, result_test))
        # with open(filename, 'a') as f:
        #     f.write("%f %f %f %f" % (train_loss, result_train, result_val, result_test))
        #     f.write("\n")

        train_curve.append(result_train)
        val_curve.append(result_val)
        test_curve.append(result_test)

        scheduler.step(result_val)

        print("")

        if optimizer.param_groups[0]['lr'] < args.lr_limit:
            break

    print("===== Final result: %f" % test_curve[-1])


if __name__ == '__main__':
    main()
