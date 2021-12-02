#!/usr/bin/env python3
import os
import sys

import torch

from svdnet.models import SVDNet
from svdnet.datasets import apply_predictor
from svdnet.utils import load_model


TRAIN_DIR = "../Market-1501-v15.09.15/bounding_box_train/"
TEST_DIR = "../Market-1501-v15.09.15/bounding_box_test/"
QUERY_DIR = "../Market-1501-v15.09.15/query/"
CHECKPOINT_DIR = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    model_svd = SVDNet(1, eval_only=True)
    model_svd.cuda()

    load_model(model_svd, CHECKPOINT_DIR)

    gallery_feats, targets, _ = apply_predictor(lambda x: model_svd.get_feat_vector(x), TEST_DIR, batch_size=16)
    model_svd.set_gallery(gallery_feats)

    predictions, gt, _ = apply_predictor(model_svd, QUERY_DIR, batch_size=16)

    accuracy = (targets[predictions[:, 0]] == gt).sum() / gt.shape[0]
    print(f"TOP-1 predicted accuracy is: {accuracy}")


if __name__ == "__main__":
    main()