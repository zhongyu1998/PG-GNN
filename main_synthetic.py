import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from dgl.data.utils import load_graphs

from utils.process import load_synthetic, separate_synthetic
from models.groupcnn import GroupCNN

torch.backends.cudnn.enabled = False


def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    pbar = tqdm(range(0, len(train_graphs), args.batch_size), unit='batch')

    loss_all = 0
    for iteration, _ in enumerate(pbar):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]

        output = model(batch_graph)
        labels = torch.FloatTensor([label for graph in batch_graph for label in graph.label]).view_as(output).to(device)
        criterion = nn.MSELoss()

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

    output = pass_data_iteratively(model, graphs)
    labels = torch.FloatTensor([label for graph in graphs for label in graph.label]).view_as(output).to(device)
    metric = nn.L1Loss(reduction='mean')
    mae = metric(output, labels).cpu().item()

    return mae


def main():
    # Parameters settings
    # Note: Hyper-parameters need to be tuned to obtain the results reported in the paper.
    #       Please refer to our paper for more details about hyper-parameter configurations.
    parser = argparse.ArgumentParser(description='PyTorch implementation of PG-GNN for synthetic datasets')
    parser.add_argument('--dataset', type=str, default="ER",
                        help='name of dataset: ER or regular (default: ER)')
    parser.add_argument('--task', type=str, default="triangle",
                        help='task type of substructure counting: triangle or clique (default: triangle)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=800,
                        help='maximum number of training epochs (default: 800)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='reduce factor of learning rate (default: 0.5)')
    parser.add_argument('--lr_patience', type=int, default=20,
                        help='decay rate patience of learning rate (default: 20)')
    parser.add_argument('--lr_limit', type=float, default=5e-6,
                        help='minimum learning rate, stop training once it is reached (default: 5e-6)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for running the experiment (default: 2)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP/RNN EXCLUDING the input one (default: 2)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.0,
                        help='dropout ratio after the final layer (default: 0.0)')
    parser.add_argument('--graph_pooling_type', type=str, default="none", choices=["none", "sum", "average"],
                        help='pooling for all nodes in a graph: none, sum, or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="lstm", choices=["srn", "gru", "lstm"],
                        help='pooling for neighboring nodes: srn, gru, or lstm')
    args = parser.parse_args()

    # set up seeds and gpu device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data, all_labels = load_graphs('synthetic/%s.bin' % args.dataset)
    graphs = load_synthetic(data, args.task)
    train_graphs, val_graphs, test_graphs = separate_synthetic(graphs, args.dataset)

    model = GroupCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim,
                     1, args.final_dropout, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=args.lr_factor,
                                                     patience=args.lr_patience,
                                                     verbose=True)

    train_curve = []
    val_curve = []
    test_curve = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_graphs, optimizer, epoch)
        mae_train = evaluate(args, model, device, train_graphs)
        mae_val = evaluate(args, model, device, val_graphs)
        mae_test = evaluate(args, model, device, test_graphs)

        print("MAE train: %f, val: %f, test: %f" % (mae_train, mae_val, mae_test))
        # with open(filename, 'a') as f:
        #     f.write("%f %f %f %f" % (train_loss, mae_train, mae_val, mae_test))
        #     f.write("\n")

        train_curve.append(mae_train)
        val_curve.append(mae_val)
        test_curve.append(mae_test)

        scheduler.step(mae_val)

        print("")

        if optimizer.param_groups[0]['lr'] < args.lr_limit:
            break

    print("===== Final MAE: %f" % test_curve[val_curve.index(min(val_curve))])


if __name__ == '__main__':
    main()
