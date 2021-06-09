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
import logging
from tqdm.auto import tqdm
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter

def copy(target: dict, source: dict):
    """Copies dict entries from source to target.

    Iterates over all keys in target and does a shallow copy from source.

    Args:
        target (dict): Target dict to copy to.
        source (dict): Source dict to take data from.
    """
    for name in target:
        target[name].data = source[name].data.clone()
    
def subtract_(target: dict, minuend: dict, subtrahend: dict):
    """Substract dict from dict.

    Iterates over all keys in dict and substracts shallow copies of values from each other.
    The result is saves in the target dict.

    Args:
        target (dict): Target dict to save data into.
        minuend (dict) Dict with data to be substracted from.
        subtrahend (dict): Dict with data to substract.
    """
    for name in target:
        target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()
    
def reduce_add_average(targets: list, sources: list):
    """Adds average weight-updates (dW) of all sources to target (W).

    Iterates over all targets and target keys.
    The value of each key is calculated as the mean of all corresponding values in the sources.
    The mean is added to the values in target.

    Args:
        targets (list of dict): Target dicts to add mean to.
        sources (list of dict): Source dicts to calculate mean from.
    """
    for target in targets:
        for name in target:
            tmp = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()
            target[name].data += tmp
        
def flatten(source: dict):
    """Flattens dict.

    Args:
        source (dict): Dict to flatten.

    Returns:
        Tensor: Concatenated tensor with dimension=0.
    """
    return torch.cat([value.flatten() for value in source.values()])

def select_n_random(data: torch.utils.data.Dataset, n: int=100) -> Tuple[torch.Tensor, list]:
    """Selects n random datapoints and their corresponding labels from a dataset

    Args:
        data (torch.utils.data.Dataset): Dataset to choose from.
        n (int, optional): Number of datapoints. Defaults to 100.

    Returns:
        Tuple[torch.Tensor, list]: Features, Labels
    """

    perm = torch.randperm(len(data)).numpy()
    features = torch.stack([data[i][0] for i in perm[:n]])
    labels = [data[i][1] for i in perm[:n]]
    return features, labels
        
class FederatedTrainingDevice(object):
    def __init__(self,
                model_fn: torch.nn.Module,
                data: torch.utils.data.Dataset,
                optimizer_fn: str,
                config: dict,
                writer: SummaryWriter=None,
                logger: logging.Logger=logging.getLogger(__name__),
                name: str='generic'
            ):
        """Base class for devices used in clustered federated learning.

        Args:
            model_fn (torch.nn.Module): Network model class
            data (torch.utils.data.Dataset): Dataset
            optimizer_fn (str): Optimizer name
            config (dict): Config dict containing alls configurations
            writer (SummaryWriter, optional): TensorBoard object. Defaults to None.
            logger (logging.Logger, optional): Logging object. Defaults to logging.getLogger(__name__).
            name (str, optional): Name of device (e.g. client_1, server). Defaults to 'generic'.
        """

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = name

        self.logger = logger

        self.data = data
        D_in = data[0][0].size()
        self.logger.debug("Feature size of %s: %s" % (self.name, str(D_in)))
        D_out = self.config['client']['num_agg_clusters']
        if 'dropout' in self.config[self.name]:
            self.model = model_fn(D_in, D_out, dropout=self.config[self.name]['dropout']).to(self.device)
        else:
            self.model = model_fn(D_in, D_out).to(self.device)
        self.writer = writer

        if 'momentum' not in config[self.name]:
            config[self.name]['momentum'] = config['server']['momentum']
        if 'patience' not in config[self.name]:
            config[self.name]['patience'] = config['server']['patience']

        if optimizer_fn == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config[self.name]['learning_rate'], momentum=config[self.name]['momentum'])
        elif optimizer_fn == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config[self.name]['learning_rate'])

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.config[self.name]['patience'], verbose=True)

        self.epoch = 0
        self.loss = None

        self.W = {key : value for key, value in self.model.named_parameters()}
        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}

    def synchronize(self, server=None):
        """Copy weights from server to current object.

        Args:
            server (FederatedTrainingDevice): Server object.
        """
        if server is not None:
            copy(target=self.W, source=server.W)
        
        self.model.load_state_dict(self.W)

    def copy_old(self, server=None):
        """Copy weights from server to current object.

        Args:
            server (FederatedTrainingDevice): Server object.
        """
        if server is not None:
            copy(target=self.W_old, source=server.W)
        else:
            copy(target=self.W_old, source=self.W)

    def reset(self):
        """Reset weights to state before training cycle.
        """
        copy(target=self.W, source=self.W_old)
        # self.model.load_state_dict(self.W)

    def get_sample_data(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """Get sample of data used by class.

        Args:
            idx (int): Index of data element.

        Returns:
            tuple: Return of get_sample_data from Database used.
        """
        return self.data.get_sample_data(idx)

    def get_dW(self) -> dict:
        """Returns delta of weights from model.

        Returns:
            dict: Dict containing names and tensors of layer weights
        """
        return self.dW

    def get_W(self) -> dict:
        """Returns weights from model.

        Returns:
            dict: Dict containing names and tensors of layer weights
        """
        return self.W

    def reset_layer_zero(self):
        """Re-initialization of layer (weight, bias) given as 'layer_zero' in config with random values.
        """
        device_name = self.name.split('_')[0]
        if "layer_zero" in self.config[device_name]:
            for name, param in self.model.named_parameters():
                if name.split('.')[0] == self.config[device_name]['layer_zero']:
                    if name.split('.')[1] == 'weight':
                        torch.nn.init.xavier_uniform_(param.data)
                    elif name.split('.')[1] == 'bias':
                        param.data.zero_()
                    self.logger.info("Resetting parameter device %s: %s" %(self.name, name))

    def freeze_layers(self):
        """Freezes all layer weights except from layer named by 'layer' in config.
        """
        # Get name of device e.g. client or server
        device_name = self.name.split('_')[0]
        # Check if layer is given for device (e.g. client, server) in the config file
        if "layer" in self.config[device_name]:
            # Search all layers
            for name, param in self.model.named_parameters():
                # Allow training of layer stated in config
                if name.split('.')[0] == self.config[device_name]['layer']:
                    param.requires_grad = True
                # Does not allow training of layer if not in config
                else:
                    param.requires_grad = False

        # Confirms trainable layers for debugging
        trainable_layers = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                trainable_layers.append(name)
        self.logger.debug("Trainable parameter device %s: %s" %(self.name, str(trainable_layers)))
            

    def compute_weight_update(self, epochs: int=1, loader: torch.utils.data.dataloader.DataLoader=None) -> float:
        """Performs federated learning by saving weights temporary, performing local training, saving new weights and computing delta weights.

        Args:
            epochs (int, optional): Number of epochs to train locally. Defaults to 1.
            loader (torch.utils.data.dataloader.DataLoader, optional): Dataloader for training. Defaults to None.

        Returns:
            float: Running loss normalized by samples.
        """
        self.W = {key : value for key, value in self.model.named_parameters()} # Get weights saved in model
        copy(target=self.W_old, source=self.W) # Save weights for history
        train_stats = self.train(epochs=epochs, train_loader=loader) # Perfrom training
        self.W = {key : value for key, value in self.model.named_parameters()} # Save new weights
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old) # Compute delta weights
                        
        return train_stats

    def train(self,
                epochs: int=1, logger: logging.Logger=logging.getLogger(__name__),
                train_loader: torch.utils.data.dataloader.DataLoader=None, eval_loader: torch.utils.data.dataloader.DataLoader=None
            ) -> float:
        """Train device for epochs.

        Args:
            epochs (int, optional): Number of epochs to train. Defaults to 1.
            logger (logging.Logger, optional): Logger object. Defaults to logging.getLogger(__name__).
            train_loader (torch.utils.data.dataloader.DataLoader, optional): Dataloader for training data. Defaults to None.
            eval_loader (torch.utils.data.dataloader.DataLoader, optional): Dataloader for evaluation data. Defaults to None.

        Returns:
            float: Running loss normalized by samples.
        """
        running_loss, samples = 0.0, 0
        logger.debug('Training for %d epochs' % (epochs))

        # Return if no training should be performed
        if epochs == 0:
            return 0

        # Iterate over epochs
        tk0 = tqdm(range(epochs), leave=False, total=epochs, desc='Epoch')
        for i, ep in enumerate(tk0):
            self.epoch += 1

            # Perform traning
            running_loss, correct, samples, train_f1 = self.model.train_op(self.model, self.train_loader if not train_loader else train_loader, self.optimizer)
            # Calculate running loss
            train_loss = running_loss / samples
            # Log results in console
            logger.debug('Training loss of epoch %d: %f' % (self.epoch, train_loss))
            logger.debug('Training accuracy of epoch %d: %f' % (self.epoch, correct/samples))
            # Log results in TensorBoard if given
            if self.writer is not None:
                log_string = "%s/Training/Loss" % (self.name)
                self.writer.add_scalar(log_string,
                            train_loss,
                            self.epoch * len(self.train_loader))
                
                log_string = "%s/Training/Accuracy" % (self.name)
                self.writer.add_scalar(log_string,
                            correct / samples,
                            self.epoch * len(self.train_loader))

                log_string = "%s/Training/F1-Score" % (self.name)
                self.writer.add_scalar(log_string,
                            train_f1 / samples,
                            self.epoch * len(self.train_loader))

            # Evaluate traning
            eval_loss, correct, eval_samples, eval_f1 = self.model.eval_op(self.model, self.eval_loader if not eval_loader else eval_loader)

            # Log results in console
            val_loss = eval_loss / eval_samples
            val_acc = correct / eval_samples
            val_f1 = eval_f1 / eval_samples
            logger.debug('Evaluation loss of epoch %d: %f' % (self.epoch, val_loss))
            logger.debug('Evaluation accuracy of epoch %d: %f' % (self.epoch, val_acc))
            logger.debug('Evaluation F1-Score of epoch %d: %f' % (self.epoch, val_f1))

            # Log results in TensorBoard if given
            if self.writer is not None:
                log_string = "%s/Validation/Loss" % (self.name)
                self.writer.add_scalar(log_string,
                            val_loss,
                            self.epoch * len(self.eval_loader))

                log_string = "%s/Validation/Accuracy" % (self.name)
                self.writer.add_scalar(log_string,
                            val_acc,
                            self.epoch * len(self.eval_loader))

                log_string = "%s/Validation/F1-Score" % (self.name)
                self.writer.add_scalar(log_string,
                            val_f1,
                            self.epoch * len(self.eval_loader))

            # Step for LR Scheduler
            self.scheduler.step(val_loss)
            logger.debug('Learning rate of epoch %d: %f' % (self.epoch, self.optimizer.param_groups[0]['lr']))

            # Log LR in TensorBoard if given
            if self.writer is not None:
                log_string = "%s/Learning Rate" % (self.name)
                self.writer.add_scalar(log_string,
                                self.optimizer.param_groups[0]["lr"],
                                self.epoch * len(self.train_loader))

            self.loss = val_loss
            # Save model to file at each epoch, if activated in config
            if self.name in self.config and self.config[self.name]['save/load'] == True and self.epoch % 1 == 0:
                # Also save as separate file for every 'save_epochs'
                if self.epoch % self.config[self.name]['save_epochs'] == 0:
                    if "model_name" in self.config[self.name]:
                        model_path = os.path.join(self.config['paths']['experiments-dir'], self.config[self.name]['model_name'] + "_" + str(self.epoch) + ".pth")
                    else:
                        model_path = os.path.join(self.config['paths']['experiments-dir'], self.name + "_" + str(self.epoch) + ".pth")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    logger.debug('Saving model (ep=%d, loss=%f): %s' % (self.epoch, self.loss,  model_path))
                    torch.save({
                        'epoch': self.epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': self.loss
                        }, model_path)
                    
                if "model_name" in self.config[self.name]:
                    model_path = os.path.join(self.config['paths']['experiments-dir'], self.config[self.name]['model_name'] + ".pth")
                else:
                    model_path = os.path.join(self.config['paths']['experiments-dir'], self.name + ".pth")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                logger.debug('Saving model (ep=%d, loss=%f): %s' % (self.epoch, self.loss, model_path))
                torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': self.loss
                    }, model_path)

            tk0.set_postfix(loss=(running_loss / samples))

        return running_loss / samples

    def evaluate(self, loader: torch.utils.data.dataloader.DataLoader=None) -> dict:
        """Evaluate device model performance

        Args:
            loader (torch.utils.data.dataloader.DataLoader, optional): Dataloader for evaluation. Defaults to None.

        Returns:
            dict: Contains trained epochs, accuracy (if supervised model), f1-score (if supervised model), and loss.
        """

        # Perform evaluation
        eval_loss, correct, eval_samples, eval_f1 = self.model.eval_op(self.model, self.eval_loader if not loader else loader)
        # Normalize loss by samples
        val_loss = eval_loss / eval_samples
        # Write stats to TensorBoard if given
        if self.writer is not None:
            log_string = "%s/Evaluation/Accuracy" % (self.name)
            self.writer.add_scalar(log_string,
                            correct / eval_samples,
                            self.epoch * len(self.eval_loader if not loader else loader))
            log_string = "%s/Evaluation/F1-Score" % (self.name)
            self.writer.add_scalar(log_string,
                            eval_f1 / eval_samples,
                            self.epoch * len(self.eval_loader if not loader else loader))

        self.loss = val_loss
        # Save model if activated in config
        if self.name in self.config and self.config[self.name]['save/load'] == True and self.epoch % self.config[self.name]['save_epochs'] == 0:
            if "model_name" in self.config[self.name]:
                model_path = os.path.join(self.config['paths']['experiments-dir'], self.config[self.name]['model_name'] + ".pth")
            else:
                model_path = os.path.join(self.config['paths']['experiments-dir'], self.name + ".pth")
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': self.loss
                }, model_path)

        eval_stats = {}

        eval_stats['hparam/epoch'] = self.epoch
        eval_stats['hparam/accuracy'] = correct / eval_samples
        eval_stats['hparam/f1'] = eval_f1 / eval_samples
        eval_stats['hparam/loss'] = self.loss

        return eval_stats


        