import random
import wandb

import numpy as np
import torch

from train import *
from test import *                  
from OUR_utils.utils import *     #Contains the functions: (make(), )
from tqdm.auto import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_pipeline(cfg:dict) -> None:
    # tell wandb to get started
    with wandb.init(project="Fastext model", config=cfg):
      wandb.run.name = 'SGD Trial'
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config)

      # and use them to train the model
      train(model, train_loader, criterion, optimizer, config)

      # and test its final performance
      test(model, test_loader)

    return model

if __name__ == "__main__":
    wandb.login()

    config = dict(
        epochs=5,
        classes=28,
        batch_size=256,
        learning_rate=0.001,
        input_size=256,
        dim = 256,
        depth = 2,
        heads = 4,
        mlp_dim = 512,
        dataset="Con-Text dataset",
        architecture="ConTextTransformer")
    
    model = model_pipeline(config)