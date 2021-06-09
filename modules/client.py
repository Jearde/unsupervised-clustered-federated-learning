""" 
Code adapted from:
Title: Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints
Author: Felix Sattler, Klaus-Robert Müller, Wojciech Samek
Date: 11.08.20
Code version: 23a1c38
Availability: https://github.com/felisat/clustered-federated-learning
"""
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple
from modules.device import * 

class Client(FederatedTrainingDevice):
    def __init__(self,
            model_fn: torch.nn.Module,
            optimizer_fn: str,
            data: torch.utils.data.Dataset,
            idnum: int,
            config: dict=None,
            writer: torch.utils.tensorboard.SummaryWriter=None,
            logger: logging.Logger=logging.getLogger(__name__)
        ):
        """Init for client object

        Args:
            model_fn (torch.nn.Module): Model class function
            optimizer_fn (string): Optimizer function
            data (Dataset): Dataset of data
            idnum (int): ID of node
            config (dict, optional): Dictionary conting the configuration. Defaults to None.
            writer (torch.utils.tensorboard.SummaryWriter, optional): Tensorboard object for plotting data. Defaults to None.
            logger (logging.Logger, optional): Logger object. Defaults to logging.getLogger(__name__).
        """
        super().__init__(model_fn, data, optimizer_fn, config=config, writer=writer, logger=logger, name='client') 

        torch.manual_seed(self.config['seeds']['random-seed'])
            
        self.data = data

        n_train = int(min( len(data)*self.config['client']['train_frac'], self.config['client']['max_data'] ))
        n_eval = int(np.ceil(min( len(data) - n_train, self.config['client']['max_data']*(1-self.config['client']['train_frac']) )))
        n_rest = len(data) - n_train
        data_train, data_rest = torch.utils.data.random_split(self.data, [n_train, n_rest])
        n_rest = len(data_rest) - n_eval
        data_eval, data_rest = torch.utils.data.random_split(data_rest, [n_eval, n_rest])

        self.train_loader = DataLoader(data_train, batch_size=self.config['client']['batch_size'], shuffle=self.config['client']['shuffle_train'])
        self.eval_loader = DataLoader(data_eval, batch_size=self.config['client']['batch_size'], shuffle=self.config['client']['shuffle_test'])
        
        self.id = idnum
        self.name = "client_%02d" % (self.id)

        logger.info("Dataset: Train %d \t Eval %d" % (n_train, n_eval))

    def get_dominant_source(self) -> Tuple[int, dict]:
        """Returns dominant sound source of client

        Returns:
            [int, dict]: Index of dominant sound source, dict containing: gender, source name, and speaker ID of dominant sound source
        """
        return self.data.dominant, self.data.combinations[self.data.dominant]

    def get_ground_truth_cluster(self) -> int:
        """Returns ground truth cluster ID

        Returns:
            int: ID of source where client is in critical distant or 'B' 
        """
        return self.data.ground_truth

    def get_position(self) -> list:
        """Returns client position

        Returns:
            [float, float, float]: 3D position data of client
        """
        return self.data.position

    def predict(self, loader: torch.utils.data.dataloader.DataLoader=None, data: torch.utils.data.Dataset=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prediction method. Uses self.eval_loader if no parameter given.

        Args:
            loader (torch.utils.data.dataloader.DataLoader, optional): Dataloader used for prediction. Defaults to None.
            data (torch.utils.data.Dataset, optional): Dataset used for prediction if no loader is given. Defaults to None.

        Returns:
            [torch.Tensor, torch.Tensor, torch.Tensor]: Predicted feature, torch.Tensor if classifier model used else None, input feature
        """
        if loader != None:
            prediction, prediction_max, target = self.model.predict_op(self.model, self.eval_loader if not loader else loader)
        elif data != None:
            prediction, prediction_max, target = self.model.predict_op(self.model, data=data)
        else:
            prediction, prediction_max, target = self.model.predict_op(self.model, data=next(iter(self.eval_loader))[0][0] if not torch.is_tensor(data) else data)
        return prediction, prediction_max, target