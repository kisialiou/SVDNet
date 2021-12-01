import torch
from torch.datasets import DataLoader
import torch.optim as optim

from svdnet.models import SVDNet
from svdnet.datasets import Market_1501
from svdnet.utils import load_latest_train_model, train_model

TRAIN_DIR = "drive/MyDrive/ozon_masters/SVDNet/Market-1501-v15.09.15/bounding_box_train/"
CHECKPOINT_DIR = "drive/MyDrive/ozon_masters/SVDNet/ckpt"

# Hyperparams
LR = 0.05
MOMENTUM=0.9

def main():
    torch.manual_seed(0)

    train_dataset = Market_1501(TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    model_svd = SVDNet(train_dataset.class_count)
    optimizer = optim.Adam(model_svd.parameters(), lr=LR, momentum=MOMENTUM)

    curr_epoch = load_latest_train_model(model_svd, optimizer, CHECKPOINT_DIR, "step0")

    train_model(model_svd, train_loader, optimizer, ckpt_dir=CHECKPOINT_DIR,
                                                    start_epoch=curr_epoch, epochs_num=60)

if __name__ == "__main__":
    main()
