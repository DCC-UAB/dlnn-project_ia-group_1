import random
import wandb

import numpy as np
import torch

from train import *
from test import *                  
from Utils.utils import *     #Contains the functions: (make(), )
from tqdm.auto import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_pipeline(directory_test_train_files, images_directory, cfg:dict) -> None:
    # tell wandb to get started
    with wandb.init(project="Fastext model", config=cfg):
      wandb.run.name = 'ADAMW-SGD-MOMENTUM-BATCH_256-DRP_0.2'
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(directory_test_train_files, images_directory, config)

      # and use them to train the model
      train(model, train_loader, criterion, optimizer, config)

      # and test its final performance
      test(model, test_loader)

    return model

if __name__ == "__main__":
    wandb.login()
    directory_test_train_files = '/home/xnmaster/data/'                         #Directory containing the name of the images for the train and test set as well as the classes of each of the images
    images_directory           = '/home/xnmaster/data/JPEGImages/'              #Directory containing the images .jpg format
    config = dict(
        epochs=70,
        classes=28,
        batch_size=256,
        learning_rate=0.0001,
        input_size=256,
        dim = 256,
        depth = 4,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.2,
        dataset="Con-Text dataset",
        architecture="ConTextTransformer")
    
    model = model_pipeline(config)
    print("Done")