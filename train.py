#!/usr/bin/env python3

import os
import sys

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

sys.path.append("../")

from svdnet.models import SVDNet
from svdnet.datasets import Market_1501
from svdnet.utils import load_latest_train_model, train_model_step0, train_rri, train_more_no_rri


TRAIN_DIR = "../Market-1501-v15.09.15/bounding_box_train/"
CHECKPOINT_DIR = "../ckpt/"


os.environ["CUDA_VISIBLE_DEVICES"] = "6"


# Hyperparams
LR = 0.05
MOMENTUM=0.9

# Training params
STEP_0_EPOCHS = 53


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    train_dataset = Market_1501(TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    model_svd = SVDNet(train_dataset.class_count)
    
    curr_epoch = load_latest_train_model(model_svd, CHECKPOINT_DIR, "step_0")
    optimizer = optim.SGD(model_svd.parameters(), lr=LR, momentum=MOMENTUM)

    train_model_step0(model_svd, train_loader, optimizer, CHECKPOINT_DIR, 
                      tag="step_0", start_epoch=curr_epoch, epochs_num=STEP_0_EPOCHS)

    train_rri(model_svd, train_loader, optimizer, CHECKPOINT_DIR, rri_epochs=2)

if __name__ == "__main__":
    main()
