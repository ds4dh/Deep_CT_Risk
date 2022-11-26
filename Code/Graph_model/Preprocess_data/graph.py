"""
PyTorch and PyTorch-geometric dataset classes for graph-based data.
"""
import numpy as np
import bisect
import math
import random

import networkx as nx
from networkx.readwrite import json_graph
import json, os

import torch
import torch_geometric
assert int(torch_geometric.__version__.split('.')[0]) == 2
from torch_geometric.data import Data
from torch_geometric.data import Dataset as Dataset_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx
from sklearn.cluster import SpectralClustering


class CTDictToPyGGraph:
    """
    From a raw hierarchical dictionary loaded from `raw/json-api`, creates a torch_geometric.data.Data object.

    It uses networkx in between, and then converts it to torch geometric.

    The `exclude_fields` are the label-sensitive fields that should be excluded during training.
    """

    def __init__(self, featurizer_obj=None, exclude_fields=('StatusModule', 'ResultsSection', 'DerivedSection'),
                 root_to_leaf=True, feature_key='x'):
        self.featurizer_obj = featurizer_obj
        self.exclude_fields = exclude_fields
        self.root_to_leaf = root_to_leaf
        self.feature_key = feature_key

        self.G_raw = nx.DiGraph()

    @staticmethod
    def list_dict_to_dict_list(l_d):
        assert isinstance(l_d, list)
        assert isinstance(l_d[0], dict)

        d_l = dict()
        for l in l_d:
            for k in l.keys():
                if k not in d_l.keys():
                    d_l[k] = []
                d_l[k].append(l[k])
        return d_l

    def content_dict_to_nx_tree(self, content_dict):

        self.G_raw.clear()
        q = list(content_dict.items())
        while q:
            v, d = q.pop()
            if v in self.exclude_fields:
                if v in self.G_raw.nodes:
                    self.G_raw.remove_node(v)
                continue
            for nv, nd in d.items():
                if self.root_to_leaf:
                    self.G_raw.add_edge(v, nv)
                else:
                    self.G_raw.add_edge(nv, v)

                self.G_raw.nodes[v]['raw'] = ''

                if isinstance(nd, dict):
                    q.append((nv, nd))
                if isinstance(nv, str) and isinstance(nd, int):
                    self.G_raw.nodes[nv]['raw'] = str(nd)
                if isinstance(nd, str):
                    self.G_raw.nodes[nv]['raw'] = str(nd)
                if isinstance(nd, list) and isinstance(nd[0], list):
                    self.G_raw.nodes[nv]['raw'] = str(nd)

                if isinstance(nd, list):
                    if isinstance(nd[0], dict):
                        nd = self.list_dict_to_dict_list(nd)
                        q.append((nv, nd))

                if isinstance(nd, list):
                    if isinstance(nd[0], str):
                        self.G_raw.nodes[nv]['raw'] = str(nd)

        self.assert_has_raw()

    def assert_has_raw(self):
        # TODO:
        for n in self.G_raw.nodes():
            if 'raw' not in self.G_raw.nodes[n].keys():
                self.G_raw.nodes[n]['raw'] = ''

    def raw_to_feature(self):
        G_feat = nx.DiGraph()
        G_feat.add_nodes_from(self.G_raw)
        G_feat.add_edges_from(self.G_raw.edges)

        for n in G_feat.nodes:
            if self.featurizer_obj:
                feature = self.featurizer_obj(self.G_raw.nodes[n]['raw'])
            else:
                feature = torch.tensor([float('nan')])
            G_feat.nodes[n][self.feature_key] = feature.view(-1)

        return G_feat

    def __call__(self, content_dict):
        self.content_dict_to_nx_tree(content_dict)
        G_feat = self.raw_to_feature()
        G_pyG = from_networkx(G_feat)
        G_pyG['nodes'] = list(self.G_raw.nodes)

        return G_pyG


class DatasetCTAsGraph(Dataset_geometric):
    """
    Provides a standard torch_geometric.data.Dataset object for the collection of CT json files.

    It featurizes and stores all of the nodes and their features and the edges and saves them to disk, if not already
    done (so if the corresponding file name for each CT doesn't exist, it runs the self.process routine.)
    """
    def __init__(self, source_dir_prc, source_dir_raw=None, split_label_path=None,
                 featurizer_obj=None, feature_key='x', root_to_leaf=False,
                 exclude_fields=('StatusModule', 'ResultsSection', 'DerivedSection')):
        """
        If all the CT's within the database are pre-calculated and saved to `source_dir_prc`, there is no need
        to specify the raw directory where they came from (`source_dir_raw`) or the featurizing object used.
        """
        self.delimiter = ','
        assert source_dir_raw or split_label_path
        self.feature_key = feature_key
        self.featurizer_obj = featurizer_obj
        self.source_dir_raw = source_dir_raw
        self.source_dir_prc = source_dir_prc
        self.split_label_path = split_label_path
        self.ext = '.json'
        if self.split_label_path:
            with open(split_label_path, 'r') as f:
                lines = f.readlines()
            self.docids_list = [l.strip().split(self.delimiter)[0] for l in lines]
            # TODO: I am going to require all cases to have labels, phases and conditions. This is temporary.
            if len(lines[0].split(self.delimiter)) == 4:
                self.phases_list = [l.strip().split(self.delimiter)[1] for l in lines]
                self.conditions_list = [l.strip().split(self.delimiter)[2] for l in lines]
                self.labels_list = [int(l.strip().split(self.delimiter)[3]) for l in lines]

                self.phases_all = ['Phase 4', 'N/A', 'Phase 3', 'Phase 2', 'Phase 1']
                self.conditions_all = ['C11', 'C07', 'C25', 'C13', 'C14', 'C26', 'C19', 'C22', 'C24', 'C08',
                                       'C16', 'C21', 'C23', 'C12', 'C17', 'C10', 'C04', 'D27', 'C09', 'C06',
                                       'C18', 'C15', 'C05', 'C01', 'C20']

        else:
            self.docids_list = [_p[0:-len(self.ext)] for _p in os.listdir(self.source_dir_raw) if _p.endswith(self.ext)]

        if self.featurizer_obj:
            self.pyg_obj = CTDictToPyGGraph(self.featurizer_obj, feature_key=feature_key, root_to_leaf=root_to_leaf,
                                            exclude_fields=exclude_fields)

        super().__init__()

    @property
    def raw_dir(self):
        # For compatibility with PyG
        return self.source_dir_raw

    @property
    def processed_dir(self):
        return self.source_dir_prc

    @property
    def raw_file_names(self):
        return [_ct + self.ext for _ct in list(set(self.docids_list))]

    @property
    def processed_file_names(self):
        return [_ct + '.pt' for _ct in list(set(self.docids_list))]

    def __len__(self):
        return len(self.docids_list)

    def len(self):
        return self.__len__()

    def indices(self):
        return range(self.len())

    def process(self):
        for _i_docid, _docid in enumerate(list(set(self.docids_list))):
            print('... Processing {} ({}/{}) and saving to {}'.format(_docid, _i_docid + 1,
                                                                      len(list(set(self.docids_list))),
                                                                      self.processed_dir))
            _path = os.path.join(self.source_dir_raw, _docid + self.ext)
            with open(_path, 'r') as f:
                _doc = json.load(f)

            data = self.pyg_obj(_doc)
            data['docids'] = _docid
            assert _docid in self.processed_file_names[_i_docid]

            torch.save(data, os.path.join(self.processed_dir, self.processed_file_names[_i_docid]))

    def __getitem__(self, idx):
        docid = self.docids_list[idx]
        processed_file_name = docid + '.pt'
        data = torch.load(os.path.join(self.processed_dir, processed_file_name))
        assert docid == data['docids']

        data[self.feature_key] = torch.tensor([_e.tolist() for _e in data[self.feature_key]])

        if hasattr(self, 'labels_list'):
            data['labels'] = self.labels_list[idx]
            phase = self.phases_list[idx]
            condition = self.conditions_list[idx]
            data['phases'] = torch.tensor(
                [0 if _p != self.phases_all.index(phase) else 1 for _p in range(len(self.phases_all))])

            data['conditions'] = torch.tensor(
                [0 if _c != self.conditions_all.index(condition) else 1 for _c in range(len(self.conditions_all))])

            data['phases'] = data['phases'].unsqueeze(0)
            data['conditions'] = data['conditions'].unsqueeze(0)
            data['phases_raw'] = phase
            data['conditions_raw'] = condition

        return data


class DatasetCTAsNode(InMemoryDataset):
    """
    Considers a CT with all its nodes under a large graph, where the connection between the CT's are specified by a
    mesh_dict. So once the indiviual CT's are converted to graph objects (torch_geometric.data.Data) uing the above
    DatasetCTAsGraph, it connects the CT's (one base node that we add to represent each CT) based on the mesh_dict.

    A lot of functionalities are needed to ensure the correct enumeration of nodes w.r.t. their parent CT's and all.

    Note that this is an InMemoryDataset, so you'll need around 50GB of free RAM for that.

    """

    def __init__(self, source_path_prc, mesh_dict_path, ct_as_graph_train, ct_as_graph_valid=None,
                 ct_as_graph_test=None):
        self.source_path_prc = source_path_prc
        self.ct_as_graph_train = ct_as_graph_train
        self.ct_as_graph_valid = ct_as_graph_valid
        self.ct_as_graph_test = ct_as_graph_test

        self.mesh_dict_path = mesh_dict_path

        # TODO: These are useless to be considered as dicts.
        self.train_dict = {_docid: [] for _docid in list(set(self.ct_as_graph_train.docids_list))}
        self.valid_dict = {}
        self.test_dict = {}

        self.docids_list = list(self.ct_as_graph_train.docids_list)  # TODO: without list(), it won't be deep copy!
        if ct_as_graph_valid is not None:
            self.valid_dict = {_docid: [] for _docid in list(set(self.ct_as_graph_valid.docids_list))}
            self.docids_list += ct_as_graph_valid.docids_list
        if ct_as_graph_test is not None:
            self.test_dict = {_docid: [] for _docid in list(set(self.ct_as_graph_test.docids_list))}
            self.docids_list += ct_as_graph_test.docids_list

        self.docids_set = None  # Hell with that! The damn culprit!

        self.dim = None
        self.device = None
        self.processed_docids = []  # TODO: This is useless.

        assert len(set(self.ct_as_graph_train.docids_list).intersection(set(self.ct_as_graph_valid.docids_list))) == 0
        assert len(set(self.ct_as_graph_train.docids_list).intersection(set(self.ct_as_graph_test.docids_list))) == 0
        assert len(set(self.ct_as_graph_valid.docids_list).intersection(set(self.ct_as_graph_test.docids_list))) == 0

        self.nodes = None
        self.labels = None
        self.phases = None
        self.conditions = None

        super().__init__()

    @property
    def raw_dir(self):
        # For compatibility with PyG
        return None

    @property
    def processed_dir(self):
        return os.path.split(self.source_path_prc)[0]

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        name = os.path.split(self.source_path_prc)[1]
        assert name.endswith('.pt') or name.endswith('.pth')
        return name

    def remove_nonexisting_nodes(self, data):
        # Those nodes that are in the original mesh graph, but not labelled.
        for _node in list(set(data.nodes) - set(self.docids_list)):
            data.remove_node(_node)

        return data

    def __getitem__(self, item):
        data = torch.load(self.source_path_prc)
        self.docids_set = data.docids_set
        self.nodes = data.nodes
        self.labels = data.labels
        self.phases = data.phases
        self.conditions = data.conditions

        del data.docids_set
        del data.nodes
        del data.labels
        del data.phases
        del data.conditions

        data.node_arange = torch.arange(len(self.nodes))

        return data

    def __len__(self):
        return 1

    def indices(self):
        return range(1)

    def add_ct_to_graph(self, ct_data, data):
        # TODO: Add split tags.
        docid = ct_data.docids
        if docid not in data.labels.keys():
            data.labels[docid] = []
            data.phases[docid] = []
            data.conditions[docid] = []
        data.labels[docid].append(ct_data.labels)
        data.phases[docid].append(ct_data.phases.tolist())  # TODO: A mistake that it was a torch.tensor originally.
        data.conditions[docid].append(ct_data.conditions.tolist())
        if docid not in self.processed_docids:
            self.processed_docids.append(docid)
            nodes = ['{}_{}'.format(docid, _n) for _n in list(ct_data.nodes)]  # Prepend nodes with the docid.
            edges = ct_data.edge_index.t().tolist()
            edges = [(nodes[_e[0]], nodes[_e[1]]) for _e in edges]  # Convert indices to proper edge names.
            data.add_edges_from(edges)  # Assuming that adding edges containing new nodes automatically adds the nodes.
            data.add_edges_from([(nodes[0], docid)])  # To connect the root node to the representative CT node.
            data.features.extend(ct_data.x.tolist())
            _index = self.docids_set.index(docid)  # TODO
            data.node_map.extend([_index for _ in range(len(nodes))])
        return data

    def fetch_labels(self, node_map):
        docids = [self.docids_set[_ind.item()] for _ind in node_map]

        return torch.tensor([self.labels[_docid][0] for _docid in docids])

    def process(self):
        # TODO: Writing the next line in init was the damn culprit behind days of struggle with debugging!
        #  For some damn reason, list(set()) is something random!
        self.docids_set = sorted(list(set(self.docids_list)))
        with open(self.mesh_dict_path, 'r') as f:
            mesh_dict = json.load(f)
        data = json_graph.node_link_graph(mesh_dict)
        print('Loaded the mesh dictionary as a graph.')
        data = self.remove_nonexisting_nodes(data)
        print('Removed irrelevant nodes.')
        data = data.to_directed()
        data.add_nodes_from(self.docids_set)

        self.dim = self.ct_as_graph_train[0].num_node_features
        self.device = self.ct_as_graph_train[0].x.device
        data.features = torch.zeros(len(data.nodes), self.dim).tolist()
        data.labels = dict()
        data.phases = dict()
        data.conditions = dict()
        _ind_dict = {docid: i for i, docid in enumerate(self.docids_set)}
        data.node_map = [_ind_dict[docid] for docid in list(data.nodes)]  # TODO: Removing the list(data.nodes)
        mask_train = [1 if _docid in self.train_dict.keys() else 0 for _docid in list(data.nodes)]
        mask_valid = [1 if _docid in self.valid_dict.keys() else 0 for _docid in list(data.nodes)]
        mask_test = [1 if _docid in self.test_dict.keys() else 0 for _docid in list(data.nodes)]

        print("Extending the graph with individual CT's :")
        _num_nodes = len(list(data.nodes))
        for _i, _ct in enumerate(self.ct_as_graph_train):
            if _i % (len(self.ct_as_graph_train) // 5) == 0:
                print('{}/{} of the training set .. '.format(_i + 1, len(self.ct_as_graph_train)))
            data = self.add_ct_to_graph(_ct, data)
            self.train_dict[_ct.docids].append(tuple([_ct.phases, _ct.conditions, _ct.labels]))
        mask_train.extend([1 for _ in range(len(data.nodes) - _num_nodes)])
        mask_valid.extend([0 for _ in range(len(data.nodes) - _num_nodes)])
        mask_test.extend([0 for _ in range(len(data.nodes) - _num_nodes)])
        if self.ct_as_graph_valid is not None:
            _num_nodes = len(list(data.nodes))
            for _i, _ct in enumerate(self.ct_as_graph_valid):
                if _i % (len(self.ct_as_graph_valid) // 5) == 0:
                    print('{}/{} of the validation set ..'.format(_i + 1, len(self.ct_as_graph_valid)))
                data = self.add_ct_to_graph(_ct, data)
                self.valid_dict[_ct.docids].append(tuple([_ct.phases, _ct.conditions, _ct.labels]))
            mask_train.extend([0 for _ in range(len(data.nodes) - _num_nodes)])
            mask_valid.extend([1 for _ in range(len(data.nodes) - _num_nodes)])
            mask_test.extend([0 for _ in range(len(data.nodes) - _num_nodes)])
        if self.ct_as_graph_test is not None:
            _num_nodes = len(list(data.nodes))
            for _i, _ct in enumerate(self.ct_as_graph_test):
                if _i % (len(self.ct_as_graph_test) // 5) == 0:
                    print('{}/{} of the test set .. '.format(_i + 1, len(self.ct_as_graph_test)))
                data = self.add_ct_to_graph(_ct, data)
                self.test_dict[_ct.docids].append(tuple([_ct.phases, _ct.conditions, _ct.labels]))

            mask_train.extend([0 for _ in range(len(data.nodes) - _num_nodes)])
            mask_valid.extend([0 for _ in range(len(data.nodes) - _num_nodes)])
            mask_test.extend([1 for _ in range(len(data.nodes) - _num_nodes)])

        assert set(self.processed_docids) == set(self.docids_list)
        assert len(mask_train) == len(data.nodes)
        assert len(mask_valid) == len(data.nodes)
        assert len(mask_test) == len(data.nodes)
        assert (torch.tensor([mask_train, mask_valid, mask_test]).sum(dim=0) != 1).sum().item() == 0

        print('Converting the networkx graph to PyG graph with some useful meta-data and saving it..')
        docids_set = self.docids_set
        nodes = list(data.nodes)
        features = data.features
        labels = data.labels
        phases = data.phases
        conditions = data.conditions
        node_map = data.node_map

        data = from_networkx(data)
        data.docids_set = docids_set
        data.nodes = nodes
        data.features = torch.tensor(features)
        data.node_map = torch.tensor(node_map)
        data.mask_train = torch.tensor(mask_train)
        data.mask_valid = torch.tensor(mask_valid)
        data.mask_test = torch.tensor(mask_test)
        # data, self.slices = self.collate([data])
        data.labels = labels
        data.phases = phases
        data.conditions = conditions

        torch.save(data, self.source_path_prc)


class CTNodeSampler:
    """
    This is a customized node-sampler to be used on top of the above DatasetCTAsNode. For the moment it doesn't work,
    as some tiny adjustments should be made.

    The idea was to ensure that all the nodes corresponding to a CT are guaranteed to appear within a same mini-batch, as
    this is not guaranteed for off-the-shelf samplers.

    """
    def __init__(self, dataset_node, batch_size=5000, shuffle=False):
        self.dataset_node = dataset_node
        self.batch_size = batch_size  # Corresponding to the number of CT's, i.e., supernodes, not nodes.

        self.shuffle = shuffle
        self.docids_set = dataset_node.docids_set
        self.docids_index_map = {_docid: _i for _i, _docid in enumerate(self.docids_set)}
        node_map = self.dataset_node[0]['node_map'].tolist()
        self.supernode_node_map = {_i: [] for _i in range(len(self.docids_set))}
        for i_docid, ind in enumerate(node_map):
            self.supernode_node_map[ind].append(i_docid)

        self.edge_index = self.dataset_node[0]['edge_index']
        self.inv_edge_ind = self.get_inverted_edge_index()

        with open(dataset_node.mesh_dict_path, 'r') as f:
            mesh_dict = json.load(f)
        g_nx = json_graph.node_link_graph(mesh_dict)
        g_nx = dataset_node.remove_nonexisting_nodes(g_nx)
        g_nx.add_nodes_from(self.docids_set)
        comps = [g_nx.subgraph(c).copy() for c in sorted(nx.connected_components(g_nx), key=len, reverse=False)]
        self.bigs = {'comps': [_c for _c in comps if len(_c.nodes) >= self.batch_size]}
        self.bigs['lens'] = torch.tensor([len(_c.nodes) for _c in self.bigs['comps']])

        self.tins = {'comps': [_c for _c in comps if len(_c.nodes) < self.batch_size]}
        self.tins['lens'] = torch.tensor([len(_c.nodes) for _c in self.tins['comps']])

        self.running_idx = 0
        self.batches = [[]]

        print('The lengths of big components: ', self.bigs['lens'].tolist())
        print('The lengths of small components: ', self.tins['lens'].tolist())

        self.sample_batches()

    def get_inverted_edge_index(self):
        def get_inverted_index(arr):
            arg = np.argsort(arr)
            diff = np.ediff1d(arr[arg], to_begin=1, to_end=1)
            rep = np.repeat(np.arange(len(diff)), diff)
            inv_dict = {}
            for k in np.unique(arr):
                inv_dict[k] = arg[rep[k]:rep[k + 1]].tolist()

            return inv_dict

        inv_edge_ind = {0: get_inverted_index(self.edge_index[0].numpy()),
                        1: get_inverted_index(self.edge_index[1].numpy())}

        inv_edge_ind[0].update(
            {_ind: np.array([]) for _ind in range(len(self.dataset_node.nodes)) if _ind not in inv_edge_ind[0].keys()})
        inv_edge_ind[1].update(
            {_ind: np.array([]) for _ind in range(len(self.dataset_node.nodes)) if _ind not in inv_edge_ind[1].keys()})

        return inv_edge_ind

    def sample_batches(self):

        batches = []
        num_seen_docids = 0
        seen_big_comps = []
        seen_tin_comps = []

        while num_seen_docids < len(self.docids_set):
            print('Sampled {} supernodes from {}'.format(num_seen_docids, len(self.docids_set)))

            num_seen_bigs = torch.gather(self.bigs['lens'], -1, torch.tensor(seen_big_comps)).sum().item()
            if torch.rand(1).item() < (self.bigs['lens'].sum().item() - num_seen_bigs) / len(self.docids_set):
                _ind = random.sample([_i for _i in range(self.bigs['lens'].numel()) if _i not in seen_big_comps], 1)[0]
                seen_big_comps.append(_ind)
                _batches = self.sample_from_big_component(self.bigs['comps'][_ind])
                batches.extend(_batches)
                num_seen_docids += self.bigs['lens'][_ind].item()
                continue

            if len(seen_tin_comps) == len(self.tins['comps']):
                continue
            batch, seen_tin_comps = self.sample_from_tin_pool(seen_tin_comps)
            num_seen_docids += len(batch)
            batches.append(batch)
            break

        # assert num_seen_docids == len(self.docids_set)
        self.batches = batches

    def sample_from_tin_pool(self, seen_tin_comps):

        av_in = torch.tensor([_i for _i in range(self.tins['lens'].numel()) if _i not in seen_tin_comps])
        lens = self.tins['lens'][av_in]
        p = lens / lens.sum()
        l_bar = (p * lens).sum().item()
        num_samples = min(int(self.batch_size // l_bar), p.numel())
        _inds = av_in[p.multinomial(num_samples=num_samples, replacement=False)].tolist()
        batch = [j for i in [list(self.tins['comps'][_ind].nodes) for _ind in _inds] for j in i]
        batch = [self.docids_index_map[_docid] for _docid in batch]

        seen_tin_comps += _inds

        return batch, seen_tin_comps

    def sample_from_big_component(self, component):
        # TODO
        n_clusters = math.ceil(len(component) / self.batch_size)

        print('Clustering a big component with {} components to {} clusters..'.format(
            len(component), n_clusters))

        adj = nx.to_scipy_sparse_matrix(component)
        clusters = SpectralClustering(affinity='precomputed', assign_labels="discretize",
                                      n_clusters=n_clusters).fit_predict(adj)
        _batches = []
        for _c in range(n_clusters):
            # TODO
            _batch = [list(component.nodes)[_i] for _i in torch.where(torch.from_numpy(clusters) == _c)[0].tolist()]
            _batch = [self.docids_index_map[_docid] for _docid in _batch]
            assert len(_batch) > 0
            _batches.append(_batch)

        return _batches

    def restart(self):
        self.running_idx = 0
        if self.shuffle:
            self.sample_batches()

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return self

    def __next__(self):
        if self.running_idx < len(self):
            batch = self.batches[self.running_idx]
            self.running_idx += 1
            return self.slicer(batch)
        else:
            self.restart()
            raise StopIteration

    @staticmethod
    def multi_select_obsolete(tensor, vals_list):
        vals = torch.tensor(vals_list).view(-1, 1)
        expanded = tensor.expand(vals.shape[0], -1)
        msk = torch.any(expanded == vals, 0)
        return torch.where(msk)[0]

    @staticmethod
    def multi_select(a, b):
        """ Courtesy of https://www.followthesheep.com/?p=1366"""
        a = a.numpy()
        b = np.array(b)

        a1 = np.argsort(a)
        b1 = np.argsort(b)

        sort_left_b = b[b1].searchsorted(a[a1], side='left')
        sort_right_b = b[b1].searchsorted(a[a1], side='right')

        inds_a = (sort_right_b - sort_left_b > 0).nonzero()[0]
        return np.sort(a1[inds_a])

    def slicer(self, batch):

        inds = sorted([i for j in [self.supernode_node_map[_ind] for _ind in batch] for i in j])

        _ind0 = np.array([i for j in [self.inv_edge_ind[0][i] for i in inds] for i in j])
        _ind1 = np.array([i for j in [self.inv_edge_ind[1][i] for i in inds] for i in j])
        torch.from_numpy(np.intersect1d(_ind0, _ind1))
        sel = torch.from_numpy(np.intersect1d(_ind0, _ind1))

        edge_index = torch.index_select(self.dataset_node[0]['edge_index'], 1, sel)
        node_map = self.dataset_node[0]['node_map'][inds]
        features = self.dataset_node[0]['features'][inds, :]
        mask_train = self.dataset_node[0]['mask_train'][inds]
        mask_valid = self.dataset_node[0]['mask_valid'][inds]
        mask_test = self.dataset_node[0]['mask_test'][inds]
        node_arange = inds

        data = Data()
        data['edge_index'] = edge_index
        data['features'] = features
        data['mask_train'] = mask_train
        data['mask_valid'] = mask_valid
        data['mask_test'] = mask_test
        data['node_arange'] = node_arange

        return data

# Below are the deprecated stuff:

class DatasetCTGraph_Obsolete(Dataset_geometric):
    def __init__(self, featurizer_obj, source_dir, split_label_path, feature_key='x', root_to_leaf=False,
                 exclude_fields=('StatusModule', 'ResultsSection', 'DerivedSection')):
        self.featurizer_obj = featurizer_obj
        self.source_dir = source_dir
        with open(split_label_path, 'r') as f:
            lines = f.readlines()
        self.docids_list = [l.strip().split()[0] for l in lines]
        try:
            self.labels_list = [l.strip().split()[1] for l in lines]
        except:
            self.labels_list = [None for l in lines]

        self.pyg_obj = CTDictToPyGGraph(self.featurizer_obj, feature_key=feature_key, root_to_leaf=root_to_leaf,
                                        exclude_fields=exclude_fields)

    def __len__(self):
        return len(self.docids_list)

    def __getitem__(self, item):
        label = self.labels_list[item]
        docid = self.docids_list[item]
        doc_path = os.path.join(self.source_dir, docid + '.json')
        with open(doc_path, 'r') as f:
            doc = json.load(f)

        data = self.pyg_obj(doc)
        data['docids'] = docid
        data['labels'] = label

        return data


class DatasetCTGraphInMemory_Obsolete(InMemoryDataset):
    def __init__(self, root, dataset_base_obj, transform=None, pre_transform=None):
        self.dataset_base_obj = dataset_base_obj

        super(DatasetCTGraphInMemory_Obsolete, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['processed.pt']

    def process(self):
        data_list = []
        for idx in range(len(self.dataset_base_obj)):
            if idx % 100 == 0:
                print(idx)

            data = self.dataset_base_obj[idx]
            data_list.append(data)

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])


def nested_dict_pairs_iterator(dict_obj):
    """
    A recursion over a dictionary to get all keys.
    """

    # Iterate over all key-value pairs of dict argument
    for key, value in dict_obj.items():

        # Change list of dictionaries to dictionaries of lists:
        if isinstance(value, list):
            if isinstance(value[0], dict):
                value = CTDictToPyGGraph.list_dict_to_dict_list(value)

        # Check if value is of dict type
        if isinstance(value, dict):
            # If value is dict then iterate over all its values
            for pair in nested_dict_pairs_iterator(value):
                yield key, *pair
        else:
            # If value is not dict type then yield the value
            yield key, value


def get_node_base_map(content):
    """
    To map the nodes to parent nodes
    """
    node_base_dict = {}
    pairs = list(nested_dict_pairs_iterator(content))
    for pair in pairs:
        for p in pair[0:-1]:
            node_base_dict[p] = pair[0]

    return node_base_dict


if __name__ == "__main__":
    """
    This is to get a mapping from the nodes to the base nodes. It will be used to map the explainability from node level
    to base-node level. The idea is to make it more robust.
    """

    from collection import ROOT_DIR, RAW_DIR

    with open(os.path.join(ROOT_DIR, 'splits/raw/eligible.txt'), 'r') as f:
        nctids_list = [_line.strip() for _line in f.readlines()]

    node_base_dict = dict()
    for i_nctid, nctid in enumerate(nctids_list):
        print('{}/{}'.format(i_nctid+1, len(nctids_list)))

        with open(os.path.join(RAW_DIR, '{}.json'.format(nctid)), 'r') as f:
            content = json.load(f)['FullStudy']['Study']['ProtocolSection']

        node_base_dict = {**node_base_dict, **get_node_base_map(content)}

    with open(os.path.join(ROOT_DIR, 'meta', 'node_base_map.json'), 'w') as f:
        json.dump(node_base_dict, f, indent=2)








