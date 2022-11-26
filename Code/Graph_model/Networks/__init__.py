import sys
sys.path.insert(0, '../../')
from ML.networks.flat import CMFMLP
from ML.networks.graph import GraphCondGlobal, GraphCondSelective


def network_loader_flat(config_net):
    """
    To load the networks for flat variants.
    """
    if config_net["net"] == 'CMFMLP':
        net = CMFMLP(config_net["d_cond"], config_net["d_in"], config_net["c_in"], config_net["dropout"], config_net["n_c"])

    return net


def network_loader_graph(config_net):
    """
    To load the networks for graph variants.
    """
    if 'global' in config_net["pool"]["method"].lower():
        net = GraphCondGlobal(config_network=config_net)

    elif 'selective' in config_net["pool"]["method"].lower():
        net = GraphCondSelective(config_network=config_net)

    return net

