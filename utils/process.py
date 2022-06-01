import networkx as nx
import numpy as np
import torch

from itertools import chain, combinations
from sklearn.model_selection import StratifiedKFold


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None):
        """
            g: a networkx graph
            label: the graph label
            node_tags: a list of node tags
            node_features: a torch float tensor representing the node features
            neighbors: a list of neighbors (without self-loop)
            max_neighbor: the maximum number of neighbors
            edge_mat: a torch long tensor containing the edge list, will be used to create a torch sparse tensor
        """
        self.g = g
        self.label = label
        self.node_tags = node_tags
        self.node_features = 0
        self.neighbors = []
        self.max_neighbor = 0
        self.edge_mat = 0


def load_data(dataset, degree_as_tag):
    """
        dataset: name of dataset
        degree_as_tag: let the input node features be node degrees
    """

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if l not in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.DiGraph()
            node_tags = []
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # node attributes are unavailable
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                for k in range(2, tmp):
                    if row[k] != j:
                        g.add_edge(j, row[k])  # remove self-connection (e.g., COLLAB)

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    # add other attributes
    for g in g_list:
        g.neighbors = [[] for _ in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)

        degree_list = []
        for i in range(len(g.g)):
            degree_list.append(len(g.neighbors[i]))
        if degree_as_tag:
            g.node_tags = degree_list
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    # extract unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))
    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)


def count_node_type(train, val, test):
    tagset = set([])
    for graph in chain(train, val, test):
        tagset = tagset.union(set(graph[0].ndata['feat'].tolist()))

    return len(tagset)


def count_degree_type(train, val, test):
    degset = set([])
    for graph in chain(train, val, test):
        degset = degset.union(set(graph[0].in_degrees().tolist()))
    degset = list(degset)
    deg2index = {degset[i]: i for i in range(len(degset))}

    return deg2index


# with open('data/superpixels/%s.pkl' % dataset, 'rb') as f:
def load_cvdata(dataset, state):
    """
        dataset: training/validation/testing dataset
        state: train, val, or test
    """

    print('loading %s data' % state)
    g_list = []
    label_list = []

    for graph in dataset:
        n = graph[0].number_of_nodes()
        g = graph[0].to_networkx()
        assert len(g) == n

        l = graph[1].item()
        label_list.append(l)

        G = S2VGraph(g, l)

        G.node_features = graph[0].ndata['feat'].float()

        G.neighbors = [[] for _ in range(len(G.g))]
        for i, j in G.g.edges():
            G.neighbors[i].append(j)

        degree_list = []
        for i in range(len(G.g)):
            degree_list.append(len(G.neighbors[i]))
        G.max_neighbor = max(degree_list)

        edges = [list(pair) for pair in G.g.edges()]
        G.edge_mat = torch.LongTensor(edges).transpose(0, 1)

        g_list.append(G)

    label_set = set(label_list)
    if state == 'train':
        print('# classes: %d' % len(label_set))
    print("# %s data: %d" % (state, len(g_list)))

    return g_list, len(label_set)


# with open('data/molecules/ZINC.pkl', 'rb') as f:
def load_zincdata(dataset, num_node_type, state):
    """
        dataset: training/validation/testing dataset
        num_node_type: number of node type
        state: train, val, or test
    """

    print('loading %s data' % state)
    g_list = []
    label_list = []

    for graph in dataset:
        n = graph[0].number_of_nodes()
        g = graph[0].to_networkx()
        assert len(g) == n

        l = graph[1].item()
        label_list.append(l)

        G = S2VGraph(g, l)

        G.node_tags = graph[0].ndata['feat'].tolist()
        G.node_features = torch.zeros(len(G.node_tags), num_node_type)
        G.node_features[range(len(G.node_tags)), [tag for tag in G.node_tags]] = 1

        G.neighbors = [[] for _ in range(len(G.g))]
        for i, j in G.g.edges():
            G.neighbors[i].append(j)

        degree_list = []
        for i in range(len(G.g)):
            degree_list.append(len(G.neighbors[i]))
        G.max_neighbor = max(degree_list)

        edges = [list(pair) for pair in G.g.edges()]
        G.edge_mat = torch.LongTensor(edges).transpose(0, 1)

        g_list.append(G)

    if state == 'train':
        print('# number of node type: %d' % num_node_type)
    print("# %s data: %d" % (state, len(g_list)))

    return g_list


def load_synthetic(dataset, task):
    """
        dataset: training & validation & testing dataset
        task: task type of substructure counting
    """

    max_nodes = 0
    for graph in dataset:
        n = graph.number_of_nodes()
        if n > max_nodes:
            max_nodes = n

    print('loading data')
    g_list = []

    for graph in dataset:
        n = graph.number_of_nodes()
        g = graph.to_networkx()
        assert len(g) == n

        A = graph.adjacency_matrix(transpose=True).to_dense()
        if task == 'triangle':
            tri = A.mm(A).mul(A).mm(torch.ones(n, 1)).reshape(-1) / 2
            l = tri.tolist()
        if task == 'clique':
            l = []
            for i in range(n):
                clq = 0
                indices = torch.nonzero(A[i]).squeeze(dim=1).tolist()
                if len(indices) >= 3:
                    for c in combinations(indices, 3):
                        if A[c[0], c[1]] == 1 and A[c[0], c[2]] == 1 and A[c[1], c[2]] == 1:
                            clq = clq + 1
                l.append(clq)

        G = S2VGraph(g, l)

        G.node_tags = torch.zeros(max_nodes)
        G.node_tags[0:n] = torch.ones(n)
        G.node_features = torch.zeros(n, max_nodes)
        G.node_features[:, 0:n] = A  # torch.eye(n)

        G.neighbors = [[] for _ in range(len(G.g))]
        for i, j in G.g.edges():
            G.neighbors[i].append(j)

        degree_list = []
        for i in range(len(G.g)):
            degree_list.append(len(G.neighbors[i]))
        G.max_neighbor = max(degree_list)

        edges = [list(pair) for pair in G.g.edges()]
        G.edge_mat = torch.LongTensor(edges).transpose(0, 1)

        g_list.append(G)

    return g_list


def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed if len(graph_list) > 1000 else 0)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list


def separate_synthetic(graph_list, dataset):
    train_idx = []
    with open('synthetic/%s_train.txt' % dataset, 'r') as f:
        for line in f:
            train_idx.append(int(line.strip()))

    val_idx = []
    with open('synthetic/%s_val.txt' % dataset, 'r') as f:
        for line in f:
            val_idx.append(int(line.strip()))

    test_idx = []
    with open('synthetic/%s_test.txt' % dataset, 'r') as f:
        for line in f:
            test_idx.append(int(line.strip()))

    train_graph_list = [graph_list[i] for i in train_idx]
    val_graph_list = [graph_list[i] for i in val_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    print("# train data: %d" % len(train_graph_list))
    print("# val data: %d" % len(val_graph_list))
    print("# test data: %d" % len(test_graph_list))

    return train_graph_list, val_graph_list, test_graph_list
