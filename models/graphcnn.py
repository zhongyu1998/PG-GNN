import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("models/")
from mlp import MLP


class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim,
                 final_dropout, graph_pooling_type, neighbor_pooling_type, device):
        """
            num_layers: number of layers in GNN (INCLUDING the input layer)
            num_mlp_layers: number of layers in MLP (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units in ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio after the final linear prediction layer
            graph_pooling_type: how to aggregate all nodes in a graph (sum, average, or none)
            neighbor_pooling_type: how to aggregate neighboring nodes (sum, average, max, srn, gru, or lstm)
            device: which device to use
        """

        super(GraphCNN, self).__init__()

        self.num_layers = num_layers
        self.final_dropout = final_dropout
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.device = device

        # a list of MLPs
        self.mlps = torch.nn.ModuleList()

        # a list of batch norms applied to the output of MLP/RNN (input of the final linear prediction layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # a list of SRNs/GRUs/LSTMs
        self.rnns = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                if self.neighbor_pooling_type == "srn":
                    self.rnns.append(nn.RNN(input_dim, hidden_dim, num_layers=num_mlp_layers, batch_first=True))
                elif self.neighbor_pooling_type == "gru":
                    self.rnns.append(nn.GRU(input_dim, hidden_dim, num_layers=num_mlp_layers, batch_first=True))
                elif self.neighbor_pooling_type == "lstm":
                    self.rnns.append(nn.LSTM(input_dim, hidden_dim, num_layers=num_mlp_layers, batch_first=True))
                else:
                    pass
            else:
                if self.neighbor_pooling_type == "srn":
                    self.rnns.append(nn.RNN(hidden_dim, hidden_dim, num_layers=num_mlp_layers, batch_first=True))
                elif self.neighbor_pooling_type == "gru":
                    self.rnns.append(nn.GRU(hidden_dim, hidden_dim, num_layers=num_mlp_layers, batch_first=True))
                elif self.neighbor_pooling_type == "lstm":
                    self.rnns.append(nn.LSTM(hidden_dim, hidden_dim, num_layers=num_mlp_layers, batch_first=True))
                else:
                    pass

        # a linear function that maps the hidden representation at each layer into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

    def __preprocess_neighbors_maxpool(self, batch_graph):
        # create padded_neighbor_list in concatenated (batch) graph

        # compute the maximum number of neighbors in the current mini-batch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):  # each graph i
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):  # each node j in graph i
                # add an offset value to each neighbor index
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                # padding, dummy data is stored in -1
                pad.extend([-1] * (max_deg - len(pad)))
                # add the central node, aggregate the central node and neighboring nodes altogether
                pad.append(j + start_idx[i])
                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        # create an adjacency matrix (a block-diagonal sparse matrix) for concatenated (batch) graph

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        # add self-loops in the adjacency matrix, aggregate the central node and neighboring nodes altogether
        num_node = start_idx[-1]
        self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
        elem = torch.ones(num_node)
        Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
        Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]]))

        return Adj_block.to(self.device)

    def __preprocess_neighbors_rnnpool(self, batch_graph):
        """
            create padded_neighbor_list and neighbor_num_list in concatenated (batch) graph
            padded_neighbor_list: a list of lists containing the v-th sublist with the form of
                                  [central node v, default permutation of neighbors, central node v, paddings with -1]
            neighbor_num_list:    a list containing real lengths (sequence lengths) of
                                  [central node v, default permutation of neighbors, central node v] for different v
        """

        # compute the maximum number of neighbors in the current mini-batch
        max_deg = max([graph.max_neighbor for graph in batch_graph]) + 2  # +2: add the central node twice

        padded_neighbor_list = []
        neighbor_num_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):  # each graph i
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):  # each node j in graph i
                # add an offset value to each neighbor index
                pad = [j + start_idx[i]]  # central node
                pad.extend([n + start_idx[i] for n in graph.neighbors[j]])  # default permutation of neighbors
                pad.append(j + start_idx[i])  # central node
                pad.extend([-1] * (max_deg - len(pad)))  # padding, dummy data is stored in -1
                padded_neighbors.append(pad)
                neighbor_num_list.append(len(graph.neighbors[j]) + 2)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list), torch.LongTensor(neighbor_num_list)

    def __preprocess_graphpool(self, batch_graph):
        # create a sum or average pooling sparse matrix (num graphs x num nodes) for concatenated (batch) graph

        start_idx = [0]
        # compute the index of the starting node in each graph
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            if self.graph_pooling_type == "average":
                # average pooling
                elem.extend([1./len(graph.g)] * len(graph.g))
            else:
                # sum pooling
                elem.extend([1] * len(graph.g))
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])

        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        # element-wise minimum does not affect max-pooling
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]

        return pooled_rep

    def rnnpool(self, h, layer, padded_neighbor_list, neighbor_num_list):
        dummy = torch.zeros_like(h[0])
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pad_sequence = h_with_dummy[padded_neighbor_list]
        pack_sequence = nn.utils.rnn.pack_padded_sequence(pad_sequence, neighbor_num_list,
                                                          batch_first=True, enforce_sorted=False)
        if self.neighbor_pooling_type == "lstm":
            output, (h_n, c_n) = self.rnns[layer](pack_sequence)
        else:
            output, h_n = self.rnns[layer](pack_sequence)
        pooled_rep = h_n[-1, :, :]

        return pooled_rep

    def next_layer(self, h, layer, padded_neighbor_list=None, neighbor_num_list=None, Adj_block=None):
        # aggregate the central node and neighboring nodes altogether
        if self.neighbor_pooling_type == "max":
            # max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        elif self.neighbor_pooling_type == "srn" or self.neighbor_pooling_type == "gru" or self.neighbor_pooling_type == "lstm":
            # rnn pooling
            pooled = self.rnnpool(h, layer, padded_neighbor_list, neighbor_num_list)
        else:
            # sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        # representation after aggregation (and MLP mapping)
        if self.neighbor_pooling_type == "srn" or self.neighbor_pooling_type == "gru" or self.neighbor_pooling_type == "lstm":
            pooled_rep = pooled
        else:
            pooled_rep = self.mlps[layer](pooled)

        # batch normalization
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)

        return h

    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        graph_pool = self.__preprocess_graphpool(batch_graph)

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        elif self.neighbor_pooling_type == "srn" or self.neighbor_pooling_type == "gru" or self.neighbor_pooling_type == "lstm":
            padded_neighbor_list, neighbor_num_list = self.__preprocess_neighbors_rnnpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        # a list of hidden representations at each layer (including the input)
        hidden_rep = [X_concat]
        h = X_concat

        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max":
                h = self.next_layer(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif self.neighbor_pooling_type == "srn" or self.neighbor_pooling_type == "gru" or self.neighbor_pooling_type == "lstm":
                h = self.next_layer(h, layer, padded_neighbor_list=padded_neighbor_list, neighbor_num_list=neighbor_num_list)
            else:
                h = self.next_layer(h, layer, Adj_block=Adj_block)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling for all nodes in each graph
        for layer, h in enumerate(hidden_rep):
            if self.graph_pooling_type == "none":
                pooled_h = h
            else:
                pooled_h = torch.spmm(graph_pool, h)
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training=self.training)

        return score_over_layer
