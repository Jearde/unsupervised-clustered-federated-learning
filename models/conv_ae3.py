from cmath import nan
import numpy as np

import torch
import torch.nn.functional as F

import logging
from tqdm.auto import tqdm
from typing import Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"

def loss_function(recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Loss function

    Args:
        recon_x (torch.Tensor): Reconstructed feature
        x (torch.Tensor): Original feature

    Returns:
        torch.Tensor: Loss value
    """
    size0 = x.size()
    x = x.view(-1, size0[-1], size0[-1])
    recon_x = recon_x.view(-1, size0[-1], size0[-1])
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')
    MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
    return MSE

class ConvAE3(torch.nn.Module):
    def __init__(self, D_in: torch.Size, D_out: int):
        """Init for NN

        Args:
            D_in (torch.Size): Input Dimension
            D_out (int): Output Dimension
        """
        super(ConvAE3, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)

        self.fc1 = torch.nn.Linear(in_features=29, out_features=29)
        self.bottleneck1 = torch.nn.Linear(in_features=29, out_features=29)
        self.defc1 = torch.nn.Linear(in_features=29, out_features=29)
        
        self.deconv2 = torch.nn.ConvTranspose2d(16, 6, 5)
        self.unpool = torch.nn.MaxUnpool2d(2, 2)
        self.deconv1 = torch.nn.ConvTranspose2d(6, 1, 5)

        # self.norm1 = torch.nn.BatchNorm2d(1)
        # self.norm2 = torch.nn.BatchNorm2d(6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward path

        Args:
            x (torch.Tensor): Input feature

        Returns:
            torch.Tensor: Output feature
        """
        size0 = x.size()
        x = x.view(-1, 1, size0[-1], size0[-1])
        size1 = x.size()

        x = F.relu(self.conv1(x))
        size2 = x.size()

        x, indices1 = self.pool(x)
        size3 = x.size()

        x = F.relu(self.conv2(x))
        size4 = x.size()

        x, indices2 = self.pool(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.bottleneck1(x))
        x = F.relu(self.defc1(x))

        x = self.unpool(x, indices2, output_size=size4)
        x = F.relu(self.deconv2(x, output_size=size3))

        x = self.unpool(x, indices1, output_size=size2)
        x = self.deconv1(x, output_size=size1)
        x = torch.sigmoid(x)

        return x

    def train_op(self, model: torch.nn.Module, loader: torch.utils.data.dataloader.DataLoader, optimizer: torch.optim.Optimizer) -> Tuple[float, int, int, float]:
        """Training operation

        Args:
            model (torch.nn.Module): Model object to train
            loader (torch.utils.data.dataloader.DataLoader): Data loader
            optimizer (torch.optim.Optimizer): Optimizer used for training

        Returns:
            [float, int, int, float]: Loss, accuracy, number of samples, f1-score
        """
        model.train()
        running_loss, samples = 0.0, 0
        correct = nan
        running_f1 = nan

        desc = "Training"
        tk1 = tqdm(loader, leave=False, total=len(loader), desc=desc)
        for k, batch in enumerate(tk1):
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            loss = loss_function(model(x), x)

            running_loss += loss.item()
            # samples += y.shape[0]
            samples += 1

            loss.backward()
            # Prevent loss = NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            tk1.set_postfix(loss=(running_loss / samples))
        
        return running_loss, correct, samples, running_f1
        
    def eval_op(self, model: torch.nn.Module, loader: torch.utils.data.dataloader.DataLoader) -> Tuple[float, int, int, float]:
        """Evaluation operation

        Args:
            model (torch.nn.Module): Model object for evaluation
            loader (torch.utils.data.dataloader.DataLoader): Data loader

        Returns:
            [float, nan, int, nan]: Loss, accuracy, number of samples, f1-score
        """
        # model.train()
        model.eval()
        running_loss, samples = 0.0, 0
        correct = nan
        running_f1 = nan

        with torch.no_grad():
            tk0 = tqdm(loader, leave=False, total=len(loader), desc="Validation")
            for i, batch in enumerate(tk0):
                x, y = batch[0].to(device), batch[1].to(device)
                
                loss = loss_function(model(x), x)
                running_loss += loss.item()
                samples += 1

                tk0.set_postfix(loss=(running_loss / samples))
        
        return running_loss, correct, samples, running_f1

    def predict_op(self, model: torch.nn.Module, data: torch.utils.data.Dataset, logger: logging.Logger=logging.getLogger(__name__)) -> Tuple[torch.Tensor, None, torch.Tensor]:
        """Prediction operation

        Args:
            model (torch.nn.Module): Model object used for prediction
            data (torch.utils.data.Dataset): Data to be predicted
            logger ([type], optional): Logger object for printing evaluation loss. Defaults to logging.getLogger(__name__).

        Returns:
            [tensor, None, tensor]: Predicted feature, None, input feature
        """
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            x = data.to(device)
            pred = model(x)
            
            loss = loss_function(pred, x)
            running_loss += loss.item()
        
        logger.info('Evaluation loss: %f' % (running_loss))
        return pred.cpu(), None, x.cpu()