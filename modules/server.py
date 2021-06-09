""" 
Code adapted from:
Title: Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints
Author: Felix Sattler, Klaus-Robert Müller, Wojciech Samek
Date: 11.08.20
Code version: 23a1c38
Availability: https://github.com/felisat/clustered-federated-learning
"""
import os
import random
import torch
from torch.utils.data import DataLoader
import numpy as np
import sklearn
import sklearn.cluster
from utils.logger import plot_matrix

import sys 
# sys.path.append("modules")
from modules.device import *

class Server(FederatedTrainingDevice):
    def __init__(self, 
            model_fn: torch.nn.Module,
            optimizer_fn: str,
            data: torch.utils.data.Dataset,
            config: dict=None,
            writer: torch.utils.tensorboard.SummaryWriter=None,
            logger: logging.Logger=logging.getLogger(__name__)
        ):
        """Init of server object

        Args:
            model_fn (torch.nn.Module): Model class function
            optimizer_fn (str): Optimizer function name
            data (torch.utils.data.Dataset): Dataset of data
            config (dict, optional): Dictionary conting the configuration. Defaults to None.
            writer (torch.utils.tensorboard.SummaryWriter, optional): TensorBoard object for plotting data.. Defaults to None.
            logger (logging.Logger, optional): Logger object.. Defaults to logging.getLogger(__name__).
        """
        super().__init__(model_fn, data, optimizer_fn, config=config, writer=writer, logger=logger, name='server')

        # Set random seed to one saved in config
        torch.manual_seed(self.config['seeds']['random-seed'])
        # Set communication rounds to 0
        self.c_round = 0
            
        # Split dataset into training and evaluation data with respect to maximum data
        self.data = data
        n_train = int(min( len(data)*self.config['server']['train_frac'], self.config['server']['max_data'] ))
        n_eval = int(min( len(data) - n_train, self.config['server']['max_data']*(1-self.config['server']['train_frac']) ))
        n_rest = len(data) - n_train
        data_train, data_rest = torch.utils.data.random_split(self.data, [n_train, n_rest])
        n_rest = len(data_rest) - n_eval
        data_eval, data_rest = torch.utils.data.random_split(data_rest, [n_eval, n_rest])

        self.train_loader = DataLoader(data_train, batch_size=self.config['server']['batch_size'], shuffle=self.config['server']['shuffle_train'])
        self.eval_loader = DataLoader(data_eval, batch_size=self.config['server']['batch_size'], shuffle=self.config['server']['shuffle_test'])

        # Name of device
        self.name = "server"
        
        # Load pre-trained model from file if available
        if self.config[self.name]['save/load'] == True:
            if "model_name" in self.config[self.name]:
                model_path = os.path.join(self.config['paths']['experiments-dir'], self.config[self.name]['model_name'] + ".pth")
            else:
                model_path = os.path.join(self.config['paths']['experiments-dir'], self.name + ".pth")
            try:
                logger.info("Loading checkpoint: %s" % (str(model_path)))
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.epoch = checkpoint['epoch']
                self.loss = checkpoint['loss']
                logger.info("Loading checkpoint successful!")
            except FileNotFoundError as e:
                logger.error("Loading checkpoint failed: %s" % (str(model_path)))

        # Processing for embeddings
        # select random images and their target indices
        images, labels = select_n_random(data_train)
        images = images.view(images.shape[0], images.shape[-3], images.shape[-1], -1)

        # get the class labels for each image
        class_labels = [data.combinations[lab]['gender'] for lab in labels]
        
        # log embeddings
        features = images.view(images.shape[0], images.shape[-1], -1)
        features = images.view(features.shape[0], -1)

        # Save embeddings and model graph in TensorBoard if given
        if writer is not None:
            writer.add_embedding(features,
                                metadata=class_labels)
                                # label_img=images)

            writer.add_graph(self.model, images)
    
    def select_clients(self, clients: list, frac: float=1.0):
        """Selects random clients from given list.

        Args:
            clients (list of Client): Clients to choose from.
            frac (float, optional): Percentage of clients to return. Defaults to 1.0.

        Returns:
            list: random list of randomly chosen clients.
        """
        return random.sample(clients, int(len(clients)*frac)) 
    
    def aggregate_weight_updates(self, clients: list):
        """Adds mean weight-updates to object weights.

        All mean value over all clients weight-updates (dW) given is added to the objects weight (W).


        Args:
            clients (list of Client): List of Client objects with their weight-updates (dW).
        """
        reduce_add_average(targets=[self.W], sources=[client.dW for client in clients])

    def pairwise_angles(self, sources: list):
        """Computes cosine similarities between weight-updates.

        Args:
            sources (list of dict): List of weight-updates (dW) of sources.

        Returns:
            ndarray: Cosine similarity matrix.
        """
        angles = torch.zeros([len(sources), len(sources)])
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources):
                if "layer" in self.config[self.name] and self.config['client']['layer'] != None: # 'bottleneck1'
                    bias_name = self.config['client']['layer'] + '.bias'
                    weight_name = self.config['client']['layer'] + '.weight'
                    source1 = {bias_name: source1[bias_name], weight_name: source1[weight_name]}
                    source2 = {bias_name: source2[bias_name], weight_name: source2[weight_name]}
                s1 = flatten(source1)
                s2 = flatten(source2)
                angles[i,j] = torch.sum(s1*s2)/(torch.norm(s1)*torch.norm(s2)+1e-12)

        return angles.numpy()
        
    def compute_pairwise_similarities(self, clients: list):
        """Computes cosine similarities between weight-updates.

        Computes cosine similarities between weight-updates and saves matrix to summary writer.

        Args:
            clients (list of Client): List of client objects containing weight-updates (dW) of sources.

        Returns:
            ndarray: Cosine similarity matrix.
        """
        self.c_round += 1
        
        similarities = self.pairwise_angles([client.dW for client in clients])
        if self.writer is not None:
            self.writer.add_figure('Similarity Matrix', plot_matrix(similarities), global_step=self.c_round)
        
        return similarities
  
    def agglomerative_clustering(self, S: np.ndarray, idc: np.ndarray, num_clusters: int=None):
        """Bipartition based on similarity matrix.

        Args:
            S (ndarray): Similarities between all clients in cluster.
            idc (ndarray): Indices of clients.
            num_clusters (int, optional): Number of clusters. If None, it's set to the number in the config. Defaults to None.

        Returns:
            list of lists: Lists of lists containing client ids for each cluster.
        """
        if num_clusters == None:
            num_clusters = self.config['client']['num_agg_clusters']

        clustering = sklearn.cluster.AgglomerativeClustering(affinity="precomputed", linkage="complete", n_clusters=num_clusters).fit(-S)

        clusters = []
        for cluster in range(num_clusters):
            clusters.append([idc[i] for i in np.argwhere(clustering.labels_ == cluster).flatten()])

        return clusters
    
    def aggregate_clusterwise(self, client_clusters: list):
        """Adds mean weight-updates to client weights in clusters.

        A mean of all weight-updates (dW) of all clients in a cluster are computed and added to all clients weights (W).
        TThe mean is calculated for each cluster independently.


        Args:
            client_clusters (list of Client): List of client objects.
        """
        for cluster in client_clusters:
            reduce_add_average(targets=[client.W for client in cluster], 
                               sources=[client.dW for client in cluster])
            
            
    def compute_max_update_norm(self, cluster: list):
        """Computes maximum of normalized weight-updates.

        Args:
            cluster (list of Client): List of clients to compute from.

        Returns:
            float: Maximum normalized weight-update (dW).
        """
        max_norm = np.max([torch.norm(flatten(client.dW)).item() for client in cluster])
        if self.writer is not None:
            self.writer.add_scalar('server/Max update norm',
                            max_norm,
                            self.c_round)
        return max_norm

    
    def compute_mean_update_norm(self, cluster: list):
        """Computes normalized mean of weight-updates.

        Args:
            cluster (list of Client): List of clients to compute from.

        Returns:
            float: Mean normalized weight-update (dW).
        """
        mean_norm = torch.norm(torch.mean(torch.stack([flatten(client.dW) for client in cluster]), 
                                     dim=0)).item()
        if self.writer is not None:
            self.writer.add_scalar('server/Mean update norm',
                            mean_norm,
                            self.c_round)
        return mean_norm

    def predict(self, data: torch.utils.data.Dataset=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prediction method. Uses self.eval_loader if no parameter given.

        Args:
            data (torch.utils.data.Dataset, optional): Dataset used for prediction if no loader is given. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Predicted feature, torch.Tensor if classifier model used else None, input feature
        """
        prediction, prediction_max, target = self.model.predict_op(self.model, data=next(iter(self.eval_loader))[0][0] if not torch.is_tensor(data) else data)
        return prediction, prediction_max, target
