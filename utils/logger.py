""" 
Code adapted from:
Title: Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints
Author: Felix Sattler, Klaus-Robert Müller, Wojciech Samek
Date: 11.08.20
Code version: 23a1c38
Availability: https://github.com/felisat/clustered-federated-learning
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import logging
import os
from scipy.io import loadmat
import ezdxf # DXF file import
import itertools
import seaborn as sns
from cmath import nan
from torch.utils.tensorboard import SummaryWriter

from utils.util import *

# Define colors for RUB CD
red = np.array((198,77,50), dtype=float)  * (1/255)
green = np.array((148,193,27), dtype=float)  * (1/255)
blue = np.array((0,53,96), dtype=float)  * (1/255)
orange = np.array((228,136,65), dtype=float)  * (1/255)
gray = np.array((221,221,221), dtype=float)  * (1/255)
black = np.array((0,0,0), dtype=float)  * (1/255)

class ExperimentLogger:
    """Class for saving the federated learning stats
    """
    def log(self, values: dict):
        """Log new data

        Dict is added the object. Each key can be iterated.

        Args:
            values (dict): Dict containing the data to be logged with keys and values.
        """
        for k, v in values.items():
            if k not in self.__dict__:
                self.__dict__[k] = [v]
            else:
                self.__dict__[k] += [v]

def isnotebook() -> bool:
    """Check if python is running in IPython Notebook environment

    Returns:
        bool: True if it is running as a notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def display_train_stats(cfl_stats: ExperimentLogger, config: dict, eps_1: float, eps_2: float, eps_3: float, communication_rounds: int, writer: SummaryWriter=None):
    """Display most important UCFL stats

    Args:
        cfl_stats (ExperimentLogger): Object containing UCFL stats for all calculated rounds.
        config (dict): Configuration.
        eps_1 (float): Threshold epsilon 1.
        eps_2 (float): Threshold epsilon 2.
        eps_3 (float): Threshold epsilon 3.
        communication_rounds (int): Maximum number of communication rounds.
        writer (SummaryWriter, optional): TensorBoard instance. Defaults to None.
    """
    # Turn off info prints of ezdxf library
    logging.getLogger('ezdxf').setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    plot_idx = -1 # Plot index. Chosen to be last calculated communication round.
    c_round = cfl_stats.rounds[plot_idx] # Get current communication round
    loss_mean = np.mean(cfl_stats.loss_clients, axis=1)
    loss_std = np.std(cfl_stats.loss_clients, axis=1)

    if isnotebook():
        logger.info("Communication round: %d" % (c_round))
    else:
        logger.debug("Communication round: %d" % (c_round))
    logger.debug("Clusters: {}".format([x for x in cfl_stats.clusters[plot_idx]]))
    logger.debug("Mean norm: %f" % (cfl_stats.mean_norm[plot_idx]))
    logger.debug("Max norm: %f" % (cfl_stats.max_norm[plot_idx]))
    logger.info("Mean/Max norm: %f" % (cfl_stats.mean_norm[plot_idx]/cfl_stats.max_norm[plot_idx]))
    logger.debug("Mean loss: %f" % (loss_mean[plot_idx]))
    logger.debug("Mean std: %f" % (loss_std[plot_idx]))
    if writer is not None:
        writer.add_scalar('clustering/Mean norm : Max norm', cfl_stats.mean_norm[plot_idx]/cfl_stats.max_norm[plot_idx], c_round)
        writer.add_scalar('clustering/Mean Loss', loss_mean[plot_idx], c_round)
        writer.add_scalar('clustering/Max norm', cfl_stats.max_norm[plot_idx], c_round)
        writer.add_scalar('clustering/Mean norm', cfl_stats.mean_norm[plot_idx], c_round)
    
    # Create figure showing mean and max weight update norm as well as ratio of both
    fig = plt.figure(figsize=(16, 8))
    plt.subplot(1,2,1)
    
    plt.plot(cfl_stats.rounds, cfl_stats.mean_norm, color=red, label=r"$\|\sum_i\Delta W_i \|$")
    plt.plot(cfl_stats.rounds, cfl_stats.max_norm, color=green, label=r"$\max_i\|\Delta W_i \|$")

    plt.axhline(y=eps_1, linestyle="--", color="k", label=r"$\varepsilon_1$")
    plt.axvline(x=config['thresholds']['min_rounds'], linestyle=":", color="k", label=r"$\mathrm{min}_\tau$")

    # Indicates when a bi-partitioning happened
    if "split" in cfl_stats.__dict__:
        for s in cfl_stats.split:
            plt.axvline(x=s, linestyle="-", color=gray)

    plt.xlabel(r"Communication rounds $\tau$")
    plt.legend()
    
    plt.xlim(0, communication_rounds)

    if writer is not None:
        writer.add_figure('Rounds/Loss', fig, global_step=c_round)

    plt.subplot(1,2,2)
    
    plt.plot(cfl_stats.rounds, np.array(cfl_stats.mean_norm, dtype=np.float)/np.array(cfl_stats.max_norm, dtype=np.float), color=red, label=r"$\frac{\|\sum_i\Delta W_i \|}{\max_i\|\Delta W_i \|}$")

    plt.axhline(y=eps_3, linestyle="--", color="k", label=r"$\varepsilon_2$")

    if "split" in cfl_stats.__dict__:
        for s in cfl_stats.split:
            plt.axvline(x=s, linestyle="-", color=gray)

    plt.xlabel(r"Communication rounds $\tau$")
    plt.legend()
    
    plt.xlim(0, communication_rounds)

    if writer is not None:
        writer.add_figure('Rounds/Loss', fig, global_step=c_round)
    
    # If running in a notebook, show figure otherwise safe figure to file.
    if isnotebook():
        plt.show()
    else:
        if not os.path.exists(config['logger']['logger-dir']):
            os.makedirs(config['logger']['logger-dir'])
        plt.savefig("{}/stats_{}.jpg".format(config['logger']['logger-dir'], c_round))
        plt.close()

    # Membership values
    if 'mv_weight' in config['thresholds']:
        weight = config['thresholds']['mv_weight'] # weighting factor lambda
    else:
        weight = 0.5

    memberships, norm_intra, norm_cross = get_memberships(cfl_stats.clusters[plot_idx], cfl_stats.similarities[plot_idx], weight=weight)

    # Similarity Matrix and clusters in the room
    fig = plt.figure(figsize=(16, 8))
    # Normalized cosine similarity matrix
    plt.subplot(1,2,1)       
    plot_matrix(cfl_stats.similarities[plot_idx], fig=fig) 

    # Room with clients contining to clusters and their membership values
    plt.subplot(1,2,2)
    plot_weighted_pos_clusters(config, cfl_stats.clusters[plot_idx], weights=memberships, fig=fig, cfl_stats=cfl_stats, combinations=None)
    
    if writer is not None:
        writer.add_figure('Rounds/Clusters', fig, global_step=c_round)
    
    if isnotebook():
        plt.show()
    else:
        if not os.path.exists(config['logger']['logger-dir']):
            os.makedirs(config['logger']['logger-dir'])
        plt.savefig("{}/result_{}.jpg".format(config['logger']['logger-dir'], c_round))
        plt.close()

def plot_matrix(cm: np.ndarray, fig: matplotlib.figure.Figure=None) -> matplotlib.figure.Figure:
    """Plot a matrix with normalization

    Args:
        cm (np.ndarray): Numpy array with data
        fig (matplotlib.figure.Figure, optional): Figure to use. Defaults to None.

    Returns:
        matplotlib.figure.Figure: Resulting figure
    """
    if fig == None:
        fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Normalized cosine similarity matrix")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks, rotation=45)
    plt.yticks(tick_marks, tick_marks)

    # Normalize the confusion matrix.
    cm = np.around((cm - cm.min()) / (cm.max() - cm.min()), decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    if cm.shape[0] <= 10:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    
    plt.tight_layout()
    plt.ylabel('ID')
    plt.xlabel('ID')

    return fig

def print_face(e: ezdxf.entities.solid.Face3d):
    """Plots ground floor of AutoCAD model

    Args:
        e (ezdxf.entities.solid.Face3d): 3D Faces of AutoCAD
    """
    xx = np.array([])
    yy = np.array([])
    zz = np.array([])

    xx = np.append(xx, e.dxf.vtx0[0])
    xx = np.append(xx, e.dxf.vtx1[0])
    xx = np.append(xx, e.dxf.vtx2[0])
    xx = np.append(xx, e.dxf.vtx3[0])
    xx = np.append(xx, e.dxf.vtx0[0])

    yy = np.append(yy, e.dxf.vtx0[1])
    yy = np.append(yy, e.dxf.vtx1[1])
    yy = np.append(yy, e.dxf.vtx2[1])
    yy = np.append(yy, e.dxf.vtx3[1])
    yy = np.append(yy, e.dxf.vtx0[1])

    zz = np.append(zz, e.dxf.vtx0[2])
    zz = np.append(zz, e.dxf.vtx1[2])
    zz = np.append(zz, e.dxf.vtx2[2])
    zz = np.append(zz, e.dxf.vtx3[2])
    zz = np.append(zz, e.dxf.vtx0[2])

    gray = np.array((221,221,221), dtype=float)  * (1/255)
    plt.plot(xx, yy, color=gray, zorder=-1)
    
def plot_weighted_pos_clusters(config: dict, clusters: list, weights: list=None, 
            fig: matplotlib.figure.Figure=None, cfl_stats: ExperimentLogger=None, combinations: dict=None
        ) -> matplotlib.figure.Figure:
    """Plot 2D floor plan with clients of each cluster

    Plots the floor plan of the room 3D model with all receivers. Each receiver is color coded for their cluster.
    The saturation of the color corresponds to the membership value.

    Args:
        config (dict): Configuration
        clusters (list of list): Cluster configuration
        weights (list, optional): Membership values. Defaults to None.
        fig (matplotlib.figure.Figure, optional): Figure to use. Defaults to None.
        cfl_stats (ExperimentLogger, optional): UCFL stats. Defaults to None.
        combinations (dict, optional): Source/speaker combination. Defaults to None.

    Returns:
        matplotlib.figure.Figure: [description]
    """
    clusters = [x.tolist() if isinstance(x, np.ndarray) else x for x in clusters]
    clusters = [x if isinstance(x, list) else [x] for x in clusters]
    clusters = [x for x in clusters if x != []]

    # Add folder to config if they don't exist
    matlab_data = loadmat(config['paths']['ir-meta-dir'])
    if "ir-dir" not in config['paths']:
        config['paths']['ir-dir'] = os.path.dirname(config['paths']['ir-meta-dir'])
    if "room-dir" not in config['paths']:
        config['paths']['room-dir'] = os.path.join(os.path.dirname(config['paths']['ir-dir']), 'IN')

    # Get mean reverberation time from RIR simulation
    t30 = np.mean(matlab_data['T30_E'][:][0:-2])
    # Compute critical distance
    r_H = critical_distance(matlab_data["room_volume"][0][0], t30)

    sources = []
    for i in range(matlab_data["nsrc"][0][0]):
        sources.append((chr(matlab_data["src_ids"][0][i]) + chr(matlab_data["src_ids"][1][i])))

    if fig == None:
        fig = plt.figure(figsize=(8, 8))

    ax = plt.gca()

    # Read in AutoCAD model of room
    dxf_file = os.path.join(config['paths']['room-dir'], "MASTER.DXF")
    dxf = ezdxf.readfile(dxf_file)

    # Plot floor plan based on 3D Model faces
    msp = dxf.modelspace()
    for e in msp:
        if e.dxftype() == '3DFACE':
            print_face(e)

    name = ["Reds", "Greens", "Blues", "Purples", "Oranges", "Greys",
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            "Reds", "Greens", "Blues", "Purples", "Oranges", "Greys",
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    markers = ['o', 'D', 'd', 'P', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'X',
            'X', 'H', 'h', '*', 'p', 's', '8', '>', '<', '^', 'v', 'P', 'd', 'D', 'o']
    node_size = 70
    idx = 0
    colors = cm.rainbow(np.linspace(0,1,len(clusters)))
    for cluster, color in zip(clusters, colors):
        if hasattr(cluster, "__len__"):
            x_pos = [matlab_data["rec_positions"][0][i] for i in cluster]
            y_pos = [matlab_data["rec_positions"][1][i] for i in cluster]
            # Membership values for clients in cluster
            if weights is not None:
                c_weights = [weights[i] for i in cluster]
            else:
                c_weights = None
            # Plot clients of cluster with color and weights
            ax.scatter(x_pos, y_pos, c=c_weights, s=node_size, cmap=plt.get_cmap(name[idx]), vmin=0, vmax=1, edgecolors=black, marker=markers[idx], label="nodes of cluster $c_%d$" % (idx+1))
            # Add MV as text next to client
            for i in cluster:
                txt = matlab_data["rec_ids"][0][i]
                plt.text(matlab_data["rec_positions"][0][i]+0.1, matlab_data["rec_positions"][1][i], txt)
                if weights is not None:
                    txt = "%.2f" % (weights[i])
                    plt.text(matlab_data["rec_positions"][0][i]-0.0, matlab_data["rec_positions"][1][i]-0.2, txt, fontsize='x-small', horizontalalignment='right')
        else:
            x_pos = [matlab_data["rec_positions"][0][cluster]]
            y_pos = [matlab_data["rec_positions"][0][cluster]]
            if weights is not None:
                c_weights = [weights[cluster]]
            else:
                c_weights = None
            ax.scatter(x_pos, y_pos, c=c_weights, s=node_size, cmap=plt.get_cmap(name[idx]), vmin=0, vmax=1, marker=markers[idx], label="nodes of cluster $c_%d$" % (idx+1))
            
            txt = matlab_data["rec_ids"][0][cluster]
            plt.text(matlab_data["rec_positions"][0][cluster]+0.1, matlab_data["rec_positions"][1][cluster], txt)
            if weights is not None:
                txt = "%.2f" % (weights[cluster])
                plt.text(matlab_data["rec_positions"][0][cluster]-0.0, matlab_data["rec_positions"][1][cluster]-0.1, txt, fontsize='x-small')

        idx += 1

    # Plot sources
    x_pos = [matlab_data["src_positions"][0][i] for i in range(matlab_data["nsrc"][0][0])]
    y_pos = [matlab_data["src_positions"][1][i] for i in range(matlab_data["nsrc"][0][0])]
    ax.scatter(x_pos, y_pos, c=[black for _x in x_pos], s=.5*node_size, vmin=0, vmax=1, marker="x", label="sources")
    for i in range(matlab_data["nsrc"][0][0]):
        txt = sources[i]
        # Add source name next to it
        plt.text(matlab_data["src_positions"][0][i]+0.1, matlab_data["src_positions"][1][i], txt)
        # Add critical distance
        circle = plt.Circle((matlab_data["src_positions"][0][i], matlab_data["src_positions"][1][i]), r_H, color=green, fill=False)
        ax.add_artist(circle)

    plt.axis('equal')
    plt.tight_layout()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')

    # Plot cluster configuration or sources with their gender
    if cfl_stats != None:
        clusters = [x.tolist() if isinstance(x, np.ndarray) else x for x in clusters]
        clusters = [x if isinstance(x, list) else [x] for x in clusters]
        clusters = [x for x in clusters if x != []]
        title = str(["%s:%s" % (combination['source'], combination['gender']) for combination in cfl_stats.combinations[0].values()])
        if matlab_data["nrec"][0][0] <= 16:
            plt.text(x=plt.gca().get_xlim()[1], y=0.98*plt.gca().get_ylim()[1], ha="right", va="top", 
                    s="Clusters(%s): {}".format([x for x in clusters]) % (title))
        else:
            plt.text(x=plt.gca().get_xlim()[1], y=0.98*plt.gca().get_ylim()[1], ha="right", va="top", 
                    s="Sources: %s" % (title))
    elif combinations is not None:
        clusters = [x.tolist() if isinstance(x, np.ndarray) else x for x in clusters]
        clusters = [x if isinstance(x, list) else [x] for x in clusters]
        clusters = [x for x in clusters if x != []]
        title = str(["%s:%s" % (combination['source'], combination['gender']) for combination in combinations.values()])
        if matlab_data["nrec"][0][0] <= 16:
            plt.text(x=plt.gca().get_xlim()[1], y=0.98*plt.gca().get_ylim()[1], ha="right", va="top", 
                    s="Clusters(%s): {}".format([x for x in clusters]) % (title))
        else:
            plt.text(x=plt.gca().get_xlim()[1], y=0.98*plt.gca().get_ylim()[1], ha="right", va="top", 
                    s="Sources: %s" % (title))

    return fig

def plot_cts(cts_d: pd.DataFrame, config: dict):
    """Plots cts matrix as heatmap

    Args:
        cts_d (pd.DataFrame): Normalized cluster-to-source distance matrix.
        config (dict): Configuration.
    """
    fig = plt.figure(facecolor='w', edgecolor='k')
    plt.title("Normalized cluster-to-source distance matrix")
    sns.heatmap(cts_d.head(), annot=True, cmap=plt.cm.Blues, cbar=False)

    if isnotebook():
        plt.show()
    else:
        if not os.path.exists(config['logger']['logger-dir']):
            os.makedirs(config['logger']['logger-dir'])
        plt.savefig("{}/cts.jpg".format(config['logger']['logger-dir']))
        plt.close()